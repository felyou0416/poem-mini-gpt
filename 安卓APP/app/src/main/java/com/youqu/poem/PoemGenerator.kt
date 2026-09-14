package com.youqu.poem

import java.util.Random

/**
 * 作诗生成器（复刻 scripts/poem.py 的 gen_poems 全流程）
 *
 * 1. 特殊 token 引导: <|bos|>《种子》<|title|> → 自回归采样（遇 <|eos|> 停机）
 * 2. K 个候选 → 去重 → 平水韵评分降序 → 取前 n
 * 3. 作者行附韵部印记（〔押X韵〕）
 */
class PoemGenerator(private val model: MiniGPT, private val judge: RhymeJudge) {

    private val rnd = Random()
    private val bos get() = model.special["bos"]!!
    private val titleTok get() = model.special["title"]!!
    private val authorTok get() = model.special["author"]!!
    private val bodyTok get() = model.special["body"]!!
    private val eos get() = model.special["eos"]!!
    private val unk get() = model.special["unk"]!!

    /** 解码（复刻 poem.py _decode_v3） */
    private fun decodeV3(tokens: List<Int>): String {
        val sb = StringBuilder()
        for (t in tokens) {
            val ch = if (t in model.itos.indices) model.itos[t] else "□"
            if (ch == "<|eos|>") break
            if (ch == "<|bos|>") continue
            if (ch == "<|title|>" || ch == "<|author|>") sb.append('\n')
            else if (ch == "<|body|>") { /* 标记位，不输出字符 */ }
            else if (ch == "<|unk|>") sb.append('□')
            else sb.append(ch)
        }
        return sb.toString().trim()
    }

    /** 采样一个完整诗块（包含规范的 Prefix Prefill 预热，遇 <|eos|> 停机） */
    private fun sampleBlock(seed: String, temperature: Float): String {
        val maxNewTokens = 110
        val cache = MiniGPT.KvCache(model.nLayer, model.nHead, model.headDim)
        val tokens = ArrayList<Int>()
        tokens.add(bos)

        // 格式化题目：剥除用户多输的书名号，保证格式规范为《xxx》
        val rawSeed = seed.trim().removePrefix("《").removeSuffix("》")
        val fullTitle = "《$rawSeed》"
        for (c in fullTitle) {
            tokens.add(model.stoi[c.toString()] ?: unk)
        }
        tokens.add(titleTok)

        // 核心修复：将完整的前缀 tokens 一次性推入 KV-Cache 进行注意力预热
        var logits = FloatArray(model.vocabSize)
        for (pos in tokens.indices) {
            logits = model.forwardStep(tokens[pos], pos, cache)
        }

        // 自回归逐步采样后续内容（作者 + 正文）
        for (step in 0 until maxNewTokens) {
            if (tokens.size >= model.blockSize) break
            val nxt = model.sample(logits, temperature, 60, rnd)
            tokens.add(nxt)
            if (nxt in model.itos.indices && model.itos[nxt] == "<|eos|>") break
            logits = model.forwardStep(nxt, tokens.size - 1, cache)
        }
        return decodeV3(tokens)
    }

    /**
     * 生成 n 首诗（带细粒度进度反馈与平水韵优选重排）
     */
    fun generate(seed: String, temperature: Float, n: Int,
                 candidatesK: Int = 6, progress: (String, Int) -> Unit = { _, _ -> }): List<String> {
        val rawSeed = seed.trim().removePrefix("《").removeSuffix("》")
        if (rawSeed.isEmpty()) return emptyList()

        val candidates = ArrayList<String>()
        val targetK = maxOf(n * 2, candidatesK)
        val maxRounds = targetK + 4
        var round = 0

        while (candidates.size < targetK && round < maxRounds) {
            round++
            val pct = (candidates.size * 80 / targetK).coerceIn(5, 80)
            progress("正在构思第 $round 轮候选… (${candidates.size}/$targetK)", pct)
            val block = sampleBlock(rawSeed, temperature)
            if (block.split("\n").firstOrNull()?.startsWith("《") == true) {
                candidates.add(block)
            }
        }

        progress("正在进行平水韵格律评估与排序…", 90)

        // 去重
        val unique = LinkedHashMap<String, Unit>()
        for (b in candidates) unique.putIfAbsent(b, Unit)

        // 符号化平水韵评分排序
        val scored = unique.keys.map { b ->
            val ev = judge.scorePoem(b)
            Triple(ev.score, ev, b)
        }.sortedByDescending { it.first }

        // 组装：在作者行后附〔押X韵〕
        val result = ArrayList<String>()
        for ((_, ev, raw) in scored.take(n)) {
            val lines = raw.split("\n").map { it.trim() }.filter { it.isNotEmpty() }
            if (lines.size >= 2 && ev.rhyme != null) {
                val out = ArrayList<String>()
                for ((i, ln) in lines.withIndex()) {
                    if (i == 1) out.add("$ln  〔押${ev.rhyme}〕")
                    else out.add(ln)
                }
                result.add(out.joinToString("\n"))
            } else {
                result.add(raw)
            }
        }
        progress("创作完成", 100)
        return result
    }
}
