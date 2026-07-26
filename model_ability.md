1️⃣ 加一个“context policy”
if input_tokens < 6K:
    ctx = 8K

elif input_tokens < 12K:
    ctx = 16K

elif input_tokens < 24K:
    ctx = 32K

else:
    ❌ 不用 64K
    → 走 chunk / RAG
2️⃣ 加一个“模型能力表”
deepseek-r1:
    max_ctx: 32768
    effective_retrieval: end-only
    use_for: reasoning

gemma4:
    max_ctx: 65536
    effective_retrieval: weak
    use_for: chat

deepseek-coder:
    max_ctx: 16384
    use_for: code