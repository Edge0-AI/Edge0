"""SAVE memory settling on a fake clock and real HTTP/FIFO; no GPU or model."""
from __future__ import annotations
import io
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from serve import session_save_retry as retry
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, SessionRefused, serve
from serve import test_slots as slot_tests
from serve.test_slots import FakeEngine, FakeProc

RESULT = {'tokens': 62993, 'bytes': 1234, 'ms': 401.5}

def refused(need=355, floor=768):
    return SessionRefused('memory', f'not enough RAM to save the session ({need} MiB plus a floor of {floor} MiB needed, 503 MiB available)')

class Clock:
    def __init__(self): self.now=0.0; self.on_sleep=lambda:None
    def monotonic(self): return self.now
    def sleep(self, seconds):
        assert 0 <= seconds <= .5
        self.now += seconds
        self.on_sleep()

class Retry(unittest.TestCase):
    def setUp(self):
        self.clock=Clock()
        self.proc=SimpleNamespace(poll=lambda:None)
        self.engine=SimpleNamespace(proc=self.proc,ended=False,session_file=mock.Mock())
        self.service=SimpleNamespace(engine=self.engine)
        self.clock_patch=mock.patch.object(retry,'time',self.clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.stdout=mock.patch('sys.stdout',new_callable=io.StringIO)
        self.stdout.start(); self.addCleanup(self.stdout.stop)

    def call(self, available=5000, action='save'):
        with mock.patch.object(retry,'available_mib',side_effect=available if isinstance(available,list) else None,
                               return_value=None if isinstance(available,list) else available):
            return retry.session_file(self.service,action,'same.bin',SessionRefused)

    def test_success_does_not_poll_or_wait(self):
        self.engine.session_file.return_value=RESULT
        with mock.patch.object(retry,'available_mib',side_effect=AssertionError('unexpected memory probe')):
            self.assertEqual(retry.session_file(self.service,'save','same.bin',SessionRefused),RESULT)
        self.assertEqual(self.clock.now,0)
        self.engine.session_file.assert_called_once_with('save','same.bin')

    def test_delayed_ram_reuses_same_process_path_and_floor(self):
        self.engine.session_file.side_effect=[refused(),RESULT]
        self.assertEqual(self.call([534,541,800,1400]),RESULT)
        self.assertEqual(self.clock.now,1.5)
        self.assertIs(self.engine.proc,self.proc)
        self.assertEqual(self.engine.session_file.call_args_list,[mock.call('save','same.bin')]*2)
        self.assertEqual(retry.required_mib(refused()),1125)

    def test_persistent_pressure_expires_without_another_save(self):
        error=refused(); self.engine.session_file.side_effect=error
        with self.assertRaises(SessionRefused) as got: self.call(541)
        self.assertIs(got.exception,error)
        self.assertEqual(self.clock.now,30)
        self.assertEqual(self.engine.session_file.call_count,1)

    def test_admission_races_are_bounded_and_keep_latest_error(self):
        errors=[refused(355+n) for n in range(4)]
        self.engine.session_file.side_effect=errors
        with self.assertRaises(SessionRefused) as got: self.call()
        self.assertIs(got.exception,errors[-1])
        self.assertEqual(self.engine.session_file.call_count,4)  # first SAVE plus three retries
        self.assertEqual(self.clock.now,1)

    def test_new_refusal_rechecks_its_larger_estimate(self):
        self.engine.session_file.side_effect=[refused(),refused(2048),RESULT]
        self.assertEqual(self.call([1400,1400,2900]),RESULT)
        self.assertEqual(self.clock.now,1)

    def test_unknown_telemetry_or_estimate_never_retries(self):
        for error,available in [(refused(),None),
              (SessionRefused('memory','not enough RAM to save the session'),5000),
              (SessionRefused('memory','not enough RAM to save the session (unknown MiB plus a floor of 768 MiB needed,'),5000)]:
            with self.subTest(error=error):
                self.engine.session_file=mock.Mock(side_effect=error)
                with self.assertRaises(SessionRefused) as got: self.call(available)
                self.assertIs(got.exception,error)
                self.assertEqual(self.engine.session_file.call_count,1)

    def test_restore_other_categories_and_published_files_never_retry(self):
        for action,error in [('restore',refused()),('save',SessionRefused('io',str(refused()))),
               ('save',SessionRefused('storage',str(refused()))),('save',SessionRefused('invalid',str(refused()))),
               ('save',SessionRefused('memory',str(refused()),published=True))]:
            with self.subTest(action=action,error=error):
                self.engine.session_file=mock.Mock(side_effect=error)
                with self.assertRaises(SessionRefused) as got: self.call(action=action)
                self.assertIs(got.exception,error)
                self.assertEqual(self.engine.session_file.call_count,1)

    def test_process_death_or_replacement_ends_wait(self):
        for change in [lambda:setattr(self.proc,'poll',lambda:1),
                       lambda:setattr(self.engine,'proc',SimpleNamespace(poll=lambda:None)),
                       lambda:setattr(self.service,'engine',SimpleNamespace(proc=self.proc)),
                       lambda:setattr(self.engine,'ended',True)]:
            with self.subTest(change=change):
                self.proc=SimpleNamespace(poll=lambda:None)
                self.engine=SimpleNamespace(proc=self.proc,ended=False,session_file=mock.Mock(side_effect=refused()))
                self.service.engine=self.engine
                self.clock.now=0; self.clock.on_sleep=change
                with self.assertRaises(SessionRefused): self.call(541)
                self.assertEqual(self.engine.session_file.call_count,1)
                self.assertEqual(self.clock.now,.5)

    def test_non_admission_error_during_retry_is_returned(self):
        for error in [EngineDied('gone'), SessionRefused('io','flushing directory',published=True)]:
            self.engine.session_file=mock.Mock(side_effect=[refused(),error])
            with self.assertRaises(type(error)) as got: self.call()
            self.assertIs(got.exception,error)
            self.assertEqual(self.engine.session_file.call_count,2)

    def test_memory_probe_is_portable_and_unknown_fails_closed(self):
        for value,expected in [(1400*2**20,1400),(0,0),(-1,None),(True,None),(None,None),(float('nan'),None)]:
            with mock.patch.dict('sys.modules',{'psutil':SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=value))}):
                self.assertEqual(retry.available_mib(),expected)

class AdmissionProc(FakeProc):
    def write(self,text):
        if not self.sent and text.startswith('SAVE '):
            self.sent.append(text)
            self.engine.lines.put('SERR memory 0 '+str(refused())+'\n')
            return
        super().write(text)

class HTTP(unittest.TestCase):
    post=slot_tests.Slots.post
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        tok=ByteTokenizer(); self.engine=FakeEngine(); self.engine.session_save_reclaim=True
        self.engine.proc=AdmissionProc(self.engine)
        self.svc=Service(self.engine,tok,ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        self.svc.slot_save_path=self.directory.name
        self.httpd=serve(self.svc,port=0)
        self.base=f'http://127.0.0.1:{self.httpd.server_address[1]}'
        self.addCleanup(self.httpd.server_close); self.addCleanup(self.httpd.shutdown)

    def test_opt_in_recovers_native_refusal_and_keeps_response_and_process(self):
        proc=self.engine.proc
        with mock.patch.object(retry,'available_mib',return_value=1400):
            status,body=self.post('/slots/0?action=save',{'filename':'a.bin'})
        self.assertEqual(status,200,body)
        self.assertEqual(body,{'id_slot':0,'filename':'a.bin','n_saved':62993,'n_written':1234,'timings':{'save_ms':401.5}})
        self.assertEqual(proc.sent,[f'SAVE {Path(self.directory.name,"a.bin")}\n']*2)
        self.assertIs(self.engine.proc,proc); self.assertFalse(self.engine.ended)
        self.assertEqual(Path(self.directory.name,'a.bin').read_bytes(),b'x'*1234)

    def test_default_refusal_is_immediate_and_byte_identical(self):
        self.engine.session_save_reclaim=False
        with mock.patch.object(retry,'available_mib',side_effect=AssertionError('default must not probe')):
            status,body=self.post('/slots/0?action=save',{'filename':'off.bin'})
        self.assertEqual(status,503)
        self.assertEqual(body,{'error':{'code':503,'message':str(refused()),'type':'server_error','kind':'memory'}})
        self.assertEqual(len(self.engine.proc.sent),1)
        self.assertFalse(Path(self.directory.name,'off.bin').exists())

    def test_fifo_blocks_restore_while_save_waits_and_releases_afterwards(self):
        waiting=threading.Event(); enough=threading.Event()
        def available(): waiting.set(); return 1400 if enough.is_set() else 541
        result={}
        path=Path(self.directory.name,'fifo.bin'); path.write_bytes(b'old file')
        a=threading.Thread(target=lambda:result.update(save=self.post('/slots/0?action=save',{'filename':'fifo.bin'})))
        b=threading.Thread(target=lambda:result.update(restore=self.post('/slots/0?action=restore',{'filename':'fifo.bin'})))
        with mock.patch.object(retry,'available_mib',side_effect=available), mock.patch.object(retry,'POLL_SECONDS',.01):
            try:
                a.start(); self.assertTrue(waiting.wait(2)); b.start()
                deadline=time.monotonic()+2
                while self.svc.status['queued'] != 1 and time.monotonic()<deadline: time.sleep(.01)
                self.assertTrue(self.svc.status['busy']); self.assertEqual(self.svc.status['queued'],1)
                self.assertEqual(len(self.engine.proc.sent),1)
            finally:
                enough.set(); a.join(5)
                if b.ident is not None: b.join(5)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(result['save'][0],200); self.assertEqual(result['restore'][0],200)
        self.assertEqual(self.engine.proc.sent,[f'SAVE {path}\n',f'SAVE {path}\n',f'RESTORE {path}\n'])
        self.assertFalse(self.svc.status['busy']); self.assertEqual(self.svc.status['queued'],0)
        self.assertTrue(self.svc.fifo.acquire(blocking=False)); self.svc.fifo.release()

if __name__=='__main__': unittest.main()
