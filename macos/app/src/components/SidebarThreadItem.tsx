import { Pencil, Trash2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import type { ThreadRow } from "../chat/ipc";

export function SidebarThreadItem({
  thread,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  thread: ThreadRow;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  if (editing) {
    return (
      <li>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onRename(draft.trim() || thread.data.title);
            setEditing(false);
          }}
        >
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => setEditing(false)}
            className="w-full rounded-lg border border-accent bg-surface px-3 py-2 text-sm outline-none"
          />
        </form>
      </li>
    );
  }

  return (
    <li
      className="group flex items-center rounded-lg transition-colors hover:bg-surface2/80"
      data-active={active || undefined}
    >
      <button
        type="button"
        onClick={onSelect}
        className={
          "min-w-0 flex-1 truncate rounded-md px-2.5 py-1.5 text-left text-sm " +
          (active ? "bg-surface2 font-semibold text-fg shadow-sm" : "text-fg/85")
        }
        title={thread.data.title}
      >
        {thread.data.title}
      </button>
      <span className="mr-1 hidden shrink-0 items-center gap-0.5 group-hover:flex group-focus-within:flex">
        <button
          type="button"
          aria-label={t("chat.rename")}
          onClick={() => {
            setEditing(true);
            setDraft(thread.data.title);
          }}
          className="e0-icon-btn !size-7 !rounded-md"
        >
          <Pencil size={13} />
        </button>
        <button
          type="button"
          aria-label={t("chat.deleteThread")}
          onClick={onDelete}
          className="e0-icon-btn !size-7 !rounded-md hover:!text-err"
        >
          <Trash2 size={13} />
        </button>
      </span>
    </li>
  );
}
