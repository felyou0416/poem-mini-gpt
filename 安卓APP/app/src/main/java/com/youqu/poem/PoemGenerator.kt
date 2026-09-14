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
            val ch = model.itos[t]
            if (ch == "<|eos|>") break
            if (ch == "<|bos|>") continue
            if (ch == "<|title|>" || ch == "<|author|>") sb.append('\n')
            else if (ch == "<|body|>") { /* 空 */ }
            else if (ch == "<|unk|>") sb.append('□')
            else sb.append(ch)
        }
        return sb.toString().trim()
    }

    /** 采样一个完整诗块（复刻 _sample_block） */
    private fun sampleBlock(seed: String, temperature: Float, progress: (Int) -> Unit): String {
        val maxTokens = 140
        val cache = MiniGPT.KvCache(model.nLayer, model.nHead, model.headDim)
        val tokens = ArrayList<Int>()
        tokens.add(bos)
        for (c in "《$seed》") tokens.add(model.stoi[c.toString()] ?: unk)
        tokens.add(titleTok)

        for (step in 0 until maxTokens) {
            val tok = tokens[tokens.size - 1]
            val logits = model.forwardStep(tok, tokens.size - 1, cache)
            val nxt = model.sample(logits, temperature, 60, rnd)
            tokens.add(nxt)
            if (model.itos[nxt] == "<|eos|>") break
            if (step % 10 == 0) progress(step)
        }
        return decodeV3(tokens)
    }

    /**
     * 生成 n 首诗。
     * candidatesK: 候选数（手机端默认 6，比 PC 端 12 少，保速度）
     */
    fun generate(seed: String, temperature: Float, n: Int,
                 candidatesK: Int = 6, progress: (String) -> Unit = {}): List<String> {
        val seedClean = seed.trim()
        val candidates = ArrayList<String>()
        val targetK = maxOf(n * 4, candidatesK)
        var round = 0
        while (candidates.size < targetK && round < 6) {
            round++
            val text = sampleBlock(seedClean, temperature) { progress("第 $round 轮采样中…") }
            val block = text
            if (block.split("\n").firstOrNull()?.startsWith("《") == true) {
                candidates.add(block)
            }
        }

        // 去重
        val unique = LinkedHashMap<String, Unit>()
        for (b in candidates) unique.putIfAbsent(b, Unit)

        // 评分排序
        val scored = unique.keys.map { b ->
            val ev = judge.scorePoem(b)
            Triple(ev.score, ev, b)
        }.sortedByDescending { it.first }

        // 组装：作者行后附〔押X韵〕
        val result = ArrayList<String>()
        for ((score, ev, raw) in scored.take(n)) {
            val lines = raw.split("\n")
            var authorIndex = -1
            for ((i, ln) in lines.withIndex()) {
                if (i == 0) continue
                val s = ln.trim()
                if (authorIndex < 0 && s.isNotEmpty() && !judge.isTitleLine(s) &&
                    s.none { it in judge.linePunct } && s.length in 2..4
                ) {
                    authorIndex = i
                }
            }
            if (authorIndex > 0 && ev.rhyme != null) {
                val out = ArrayList<String>()
                for ((i, ln) in lines.withIndex()) {
                    if (i == authorIndex) out.add(ln.trimEnd() + "  〔押${ev.rhyme}〕")
                    else out.add(ln)
                }
                result.add(out.joinToString("\n").trim())
            } else {
                result.add(raw)
            }
        }
        return result
    }
}
