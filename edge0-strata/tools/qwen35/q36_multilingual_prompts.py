"""q36_multilingual_prompts.py TOKENIZER.json OUT.ids : multilingual / code chat prompts (ChatML, thinking off) as id lists, and the share of token ids
above a few draft-vocab limits in the model's own tokens (the texts below)."""
import sys, json
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
P = [
    ("en", "Explain how a hash table handles collisions, in about 150 words."),
    ("ru", "Объясни простыми словами, как работает рекурсия в программировании, и приведи пример на Python."),
    ("zh", "请用中文简单解释一下什么是机器学习，以及它和传统编程的区别。"),
    ("es", "Escribe un párrafo sobre la importancia de dormir bien y cómo afecta al rendimiento diario."),
    ("de", "Erkläre mir den Unterschied zwischen einem Prozess und einem Thread in einem Betriebssystem."),
    ("ja", "日本の四季について、それぞれの季節の特徴を簡単に説明してください。"),
    ("ar", "اشرح لي باختصار كيف تعمل الطاقة الشمسية وما هي فوائدها للبيئة."),
    ("hi", "भारत की राजधानी के बारे में कुछ वाक्य लिखिए और वहाँ के प्रसिद्ध स्थानों के नाम बताइए।"),
    ("ko", "인공지능이 일상생활에 어떤 영향을 미치는지 간단하게 설명해 주세요."),
    ("fr", "Rédige un court e-mail pour demander un rendez-vous avec un professeur au sujet d'un mémoire."),
    ("code", "Write a Python function that merges two sorted lists into one sorted list, with a short docstring and two tests."),
    ("math", "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed in m/s?"),
]
out = []
ids_all = []
for name, text in P:
    s = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tok.encode(s).ids
    out.append((name, ids))
    ids_all.append(ids)
with open(sys.argv[2], "w") as f:
    for name, ids in out:
        f.write(",".join(map(str, ids)) + "\n")
with open(sys.argv[2] + ".names", "w") as f:
    f.write("\n".join(n for n, _ in out) + "\n")
for name, ids in out:
    over = {lim: sum(1 for i in ids if i >= lim) / len(ids) for lim in (32768, 65536, 98304, 131072, 151643)}
    print("%-5s %3d tokens; share of ids >= %s" % (name, len(ids), " ".join("%d:%.0f%%" % (k, 100 * v) for k, v in over.items())))
