import pathlib,shlex,subprocess
import argparse
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--build',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
r=pathlib.Path(__file__).resolve().parent;build=a.build.resolve();a.out.mkdir(parents=True,exist_ok=True)
cmds=subprocess.check_output(['ninja','-C',str(build),'-t','commands','strata'],text=True).splitlines()
c=shlex.split(next(x for x in cmds if ' -c ' in x and str(a.source.resolve()/'sycl/src/program/generate.cpp') in x))
oldobj=c[c.index('-o')+1];obj=a.out/'gemm-stride-check.o'
c[c.index('-c')+1]=str(r/'gemm-stride-check.cpp')
for opt,val in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:c[c.index(opt)+1]=val
subprocess.run(c,cwd=build,check=True)
l=shlex.split(cmds[-1].removeprefix(': && ').removesuffix(' && :'));l[l.index('-o')+1]=str(a.out/'gemm-stride-check');l[l.index(oldobj)]=str(obj)
l=[('-Wl,--dependency-file='+str(a.out/'gemm-stride-link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in l]
subprocess.run(l,cwd=build,check=True)
