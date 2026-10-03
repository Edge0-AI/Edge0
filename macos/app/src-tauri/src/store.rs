//! Conversation store. This process is the only writer; the webview never sees SQL or file paths.

use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde::Serialize;

use edge0_core::conversation::{MessageRecord, ThreadRecord};

pub use edge0_core::conversation::{MessageStatus, Role, Usage};
use edge0_core::paths::Home;

const SCHEMA_VERSION: u32 = 1;

const MIGRATIONS: &[(u32, &[&str])] = &[(
    1,
    &[
        "CREATE TABLE IF NOT EXISTS threads(
            id         TEXT PRIMARY KEY,
            data       TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )",
        "CREATE TABLE IF NOT EXISTS messages(
            id         TEXT PRIMARY KEY,
            thread_id  TEXT NOT NULL,
            data       TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )",
        "CREATE INDEX IF NOT EXISTS idx_messages_thread_created ON messages(thread_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at)",
    ],
)];

fn wall_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadRow {
    pub id: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub data: ThreadRecord,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageRow {
    pub id: String,
    pub created_at: i64,
    pub data: MessageRecord,
}

#[derive(Debug)]
pub struct Store {
    conn: Connection,
    /// Write timestamps are `max(wall, last+1)` so `(created_at, id)` pagination is deterministic when user/assistant land in the same millisecond (ids are random UUIDs).
    clock: AtomicI64,
}

impl Store {
    pub fn open(home: &Home) -> Result<(Store, usize), String> {
        Self::open_path(&home.edge0_db())
    }

    pub fn open_path(path: &Path) -> Result<(Store, usize), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
        }
        let conn = Connection::open(path).map_err(|e| format!("failed to open conversation store: {e}"))?;
        let mut s = Store {
            conn,
            clock: AtomicI64::new(0),
        };
        s.migrate()?;
        let hi = s
            .conn
            .query_row(
                "SELECT COALESCE(MAX(x), 0) FROM (
                    SELECT created_at AS x FROM messages
                    UNION ALL SELECT created_at FROM threads
                    UNION ALL SELECT updated_at FROM threads
                 )",
                [],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0);
        s.clock.store(hi.max(wall_ms()), Ordering::Relaxed);
        let reconciled = s.reconcile_streaming()?;
        Ok((s, reconciled))
    }

    fn next_ts(&self) -> i64 {
        loop {
            let last = self.clock.load(Ordering::Relaxed);
            let next = wall_ms().max(last + 1);
            if self
                .clock
                .compare_exchange_weak(last, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return next;
            }
        }
    }

    fn user_version(&self) -> Result<u32, String> {
        self.conn
            .query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))
            .map_err(|e| e.to_string())
    }

    fn migrate(&mut self) -> Result<(), String> {
        let v = self.user_version()?;
        if v > SCHEMA_VERSION {
            return Err(format!(
                "conversation store generation {v} is newer than this app supports ({SCHEMA_VERSION}) — upgrade the app; do not guess downward"
            ));
        }
        for (ver, ddls) in MIGRATIONS {
            if *ver <= v {
                continue;
            }
            let tx = self.conn.transaction().map_err(|e| e.to_string())?;
            for ddl in *ddls {
                tx.execute_batch(ddl)
                    .map_err(|e| format!("migration {ver} failed: {e}"))?;
            }
            tx.pragma_update(None, "user_version", ver)
                .map_err(|e| e.to_string())?;
            tx.commit().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn reconcile_streaming(&mut self) -> Result<usize, String> {
        let mut rows: Vec<(String, String)> = self
            .conn
            .prepare("SELECT id, data FROM messages")
            .map_err(|e| e.to_string())?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| e.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        let mut n = 0;
        let tx = self.conn.transaction().map_err(|e| e.to_string())?;
        for (id, data) in rows.drain(..) {
            let mut rec: MessageRecord = match serde_json::from_str(&data) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if rec.status != MessageStatus::Streaming {
                continue;
            }
            rec.status = MessageStatus::Aborted;
            rec.validate()
                .map_err(|e| format!("reconciled record is invalid: {e}"))?;
            tx.execute(
                "UPDATE messages SET data = ?1 WHERE id = ?2",
                params![serde_json::to_string(&rec).map_err(|e| e.to_string())?, id],
            )
            .map_err(|e| e.to_string())?;
            n += 1;
        }
        tx.commit().map_err(|e| e.to_string())?;
        Ok(n)
    }

    // ------------------------------------------------------------------ threads

    pub fn threads(&self) -> Result<Vec<ThreadRow>, String> {
        let mut st = self
            .conn
            .prepare(
                "SELECT id, created_at, updated_at, data FROM threads ORDER BY updated_at DESC",
            )
            .map_err(|e| e.to_string())?;
        let rows = st
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for r in rows {
            let (id, created_at, updated_at, data) = r.map_err(|e| e.to_string())?;
            out.push(ThreadRow {
                id,
                created_at,
                updated_at,
                data: serde_json::from_str(&data).map_err(|e| format!("corrupt thread record: {e}"))?,
            });
        }
        Ok(out)
    }

    pub fn create_thread(&self, title: &str, model: Option<String>) -> Result<ThreadRow, String> {
        let id = format!("thr-{}", uuid::Uuid::new_v4());
        let now = self.next_ts();
        let data = ThreadRecord::new(title, model);
        self.conn
            .execute(
                "INSERT INTO threads(id, data, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
                params![id, serde_json::to_string(&data).unwrap(), now, now],
            )
            .map_err(|e| e.to_string())?;
        Ok(ThreadRow {
            id,
            created_at: now,
            updated_at: now,
            data,
        })
    }

    fn thread_data(&self, id: &str) -> Result<(String, i64), String> {
        self.conn
            .query_row(
                "SELECT data, updated_at FROM threads WHERE id = ?1",
                params![id],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
            )
            .map_err(|_| format!("thread not found: {id}"))
    }

    pub fn rename_thread(&self, id: &str, title: &str) -> Result<(), String> {
        let (data, _) = self.thread_data(id)?;
        let mut rec: ThreadRecord = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        rec.title = title.into();
        self.conn
            .execute(
                "UPDATE threads SET data = ?1, updated_at = ?2 WHERE id = ?3",
                params![serde_json::to_string(&rec).unwrap(), self.next_ts(), id],
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn set_thread_model(&self, id: &str, model: Option<String>) -> Result<(), String> {
        let (data, _) = self.thread_data(id)?;
        let mut rec: ThreadRecord = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        rec.model = model;
        self.conn
            .execute(
                "UPDATE threads SET data = ?1 WHERE id = ?2",
                params![serde_json::to_string(&rec).unwrap(), id],
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn delete_thread(&mut self, id: &str) -> Result<(), String> {
        let tx = self.conn.transaction().map_err(|e| e.to_string())?;
        tx.execute("DELETE FROM messages WHERE thread_id = ?1", params![id])
            .map_err(|e| e.to_string())?;
        tx.execute("DELETE FROM threads WHERE id = ?1", params![id])
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())
    }
}

fn touch_thread(tx: &rusqlite::Transaction<'_>, thread_id: &str, ts: i64) -> Result<(), String> {
    tx.execute(
        "UPDATE threads SET updated_at = ?1 WHERE id = ?2",
        params![ts, thread_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn write_message_in(
    tx: &rusqlite::Transaction<'_>,
    id: &str,
    rec: &MessageRecord,
) -> Result<(), String> {
    rec.validate()?;
    tx.execute(
        "UPDATE messages SET data = ?1 WHERE id = ?2",
        params![serde_json::to_string(rec).map_err(|e| e.to_string())?, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

impl Store {
    // ------------------------------------------------------------------ messages

    pub fn messages_page(
        &self,
        thread_id: &str,
        before: Option<(i64, &str)>,
        limit: u32,
    ) -> Result<Vec<MessageRow>, String> {
        let (sql, pv): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = match before {
            None => (
                "SELECT id, created_at, data FROM messages
                 WHERE thread_id = ?1 ORDER BY created_at DESC, id DESC LIMIT ?2",
                vec![Box::new(thread_id.to_string()), Box::new(limit)],
            ),
            Some((ca, id)) => (
                "SELECT id, created_at, data FROM messages
                 WHERE thread_id = ?1 AND (created_at < ?2 OR (created_at = ?2 AND id < ?3))
                 ORDER BY created_at DESC, id DESC LIMIT ?4",
                vec![
                    Box::new(thread_id.to_string()),
                    Box::new(ca),
                    Box::new(id.to_string()),
                    Box::new(limit),
                ],
            ),
        };
        let mut st = self.conn.prepare(sql).map_err(|e| e.to_string())?;
        let refs: Vec<&dyn rusqlite::types::ToSql> = pv.iter().map(|p| p.as_ref()).collect();
        let rows = st
            .query_map(refs.as_slice(), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for r in rows {
            let (id, created_at, data) = r.map_err(|e| e.to_string())?;
            out.push(MessageRow {
                id,
                created_at,
                data: serde_json::from_str(&data).map_err(|e| format!("corrupt message record: {e}"))?,
            });
        }
        out.reverse();
        Ok(out)
    }

    fn message_data(&self, id: &str) -> Result<(String, String), String> {
        self.conn
            .query_row(
                "SELECT thread_id, data FROM messages WHERE id = ?1",
                params![id],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )
            .map_err(|_| format!("message not found: {id}"))
    }

    fn write_message(&self, id: &str, rec: &MessageRecord) -> Result<(), String> {
        rec.validate()?;
        self.conn
            .execute(
                "UPDATE messages SET data = ?1 WHERE id = ?2",
                params![serde_json::to_string(rec).map_err(|e| e.to_string())?, id],
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn message_start(
        &mut self,
        thread_id: &str,
        role: Role,
        content: &str,
        model: &str,
        params_snapshot: Option<serde_json::Value>,
        id: Option<String>,
    ) -> Result<MessageRow, String> {
        let id = match id {
            Some(given) => {
                if !given.starts_with("msg-")
                    || given.len() > 80
                    || !given
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                {
                    return Err("explicit message id must use the msg- prefix and a safe character set".into());
                }
                let exists: i64 = self
                    .conn
                    .query_row(
                        "SELECT COUNT(*) FROM messages WHERE id = ?1",
                        params![given],
                        |r| r.get(0),
                    )
                    .map_err(|e| e.to_string())?;
                if exists > 0 {
                    return Err(format!("message id already exists: {given}"));
                }
                given
            }
            None => format!("msg-{}", uuid::Uuid::new_v4()),
        };
        let now = self.next_ts();
        let mut rec = MessageRecord::new(role, content, model, now);
        rec.thread_id = Some(thread_id.into());
        rec.params = params_snapshot;
        rec.validate()?;
        let tx = self.conn.transaction().map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO messages(id, thread_id, data, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, thread_id, serde_json::to_string(&rec).unwrap(), now],
        )
        .map_err(|e| e.to_string())?;
        touch_thread(&tx, thread_id, now)?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(MessageRow {
            id,
            created_at: now,
            data: rec,
        })
    }

    pub fn message_append(
        &mut self,
        id: &str,
        content_delta: Option<&str>,
        reasoning_delta: Option<&str>,
    ) -> Result<(), String> {
        let (thread_id, data) = self.message_data(id)?;
        let mut rec: MessageRecord = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        if let Some(d) = content_delta {
            rec.content.push_str(d);
        }
        if let Some(d) = reasoning_delta {
            rec.reasoning_content
                .get_or_insert_with(String::new)
                .push_str(d);
        }
        let ts = self.next_ts();
        let tx = self.conn.transaction().map_err(|e| e.to_string())?;
        write_message_in(&tx, id, &rec)?;
        touch_thread(&tx, &thread_id, ts)?;
        tx.commit().map_err(|e| e.to_string())
    }

    pub fn message_finish(
        &mut self,
        id: &str,
        status: MessageStatus,
        code: Option<&str>,
        usage: Option<Usage>,
    ) -> Result<MessageRow, String> {
        let (thread_id, data) = self.message_data(id)?;
        let mut rec: MessageRecord = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        if rec.status == MessageStatus::Streaming && status == MessageStatus::Streaming {
            return Err("streaming is not a terminal status".into());
        }
        rec.status = status;
        rec.code = code.map(str::to_string);
        if usage.is_some() {
            rec.usage = usage;
        }
        rec.validate()?;
        let ts = self.next_ts();
        let tx = self.conn.transaction().map_err(|e| e.to_string())?;
        write_message_in(&tx, id, &rec)?;
        touch_thread(&tx, &thread_id, ts)?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(MessageRow {
            id: id.into(),
            created_at: rec.created_at,
            data: rec,
        })
    }

    pub fn message_set_content(&self, id: &str, content: &str) -> Result<(), String> {
        let (_, data) = self.message_data(id)?;
        let mut rec: MessageRecord = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        rec.content = content.into();
        self.write_message(id, &rec)
    }

    pub fn thread_truncate_and_start(
        &mut self,
        thread_id: &str,
        from_created_at: i64,
        role: Role,
        content: &str,
        model: &str,
        params_snapshot: Option<serde_json::Value>,
    ) -> Result<MessageRow, String> {
        let id = format!("msg-{}", uuid::Uuid::new_v4());
        let now = self.next_ts();
        let mut rec = MessageRecord::new(role, content, model, now);
        rec.thread_id = Some(thread_id.into());
        rec.params = params_snapshot;
        rec.validate()?;
        let tx = self.conn.transaction().map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM messages WHERE thread_id = ?1 AND created_at >= ?2",
            params![thread_id, from_created_at],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO messages(id, thread_id, data, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, thread_id, serde_json::to_string(&rec).unwrap(), now],
        )
        .map_err(|e| e.to_string())?;
        touch_thread(&tx, thread_id, now)?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(MessageRow {
            id,
            created_at: now,
            data: rec,
        })
    }

    pub fn messages_delete_from(
        &mut self,
        thread_id: &str,
        from_id: &str,
    ) -> Result<usize, String> {
        let from_created_at: i64 = self
            .conn
            .query_row(
                "SELECT created_at FROM messages WHERE id = ?1 AND thread_id = ?2",
                params![from_id, thread_id],
                |r| r.get(0),
            )
            .map_err(|_| format!("message is not in this thread: {from_id}"))?;
        let ts = self.next_ts();
        let tx = self.conn.transaction().map_err(|e| e.to_string())?;
        let n = tx
            .execute(
                "DELETE FROM messages WHERE thread_id = ?1 AND created_at >= ?2",
                params![thread_id, from_created_at],
            )
            .map_err(|e| e.to_string())?;
        touch_thread(&tx, thread_id, ts)?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(n)
    }

    pub fn with_conn<T>(&self, f: impl FnOnce(&Connection) -> T) -> T {
        f(&self.conn)
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        let _ = self.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)");
    }
}
