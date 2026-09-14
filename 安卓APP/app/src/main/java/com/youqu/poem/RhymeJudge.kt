package com.youqu.poem

import android.content.Context
import org.json.JSONObject

/**
 * 平水韵格律评分器（与 scripts/rhyme_dict.py 逐规则对齐）
 *
 * base 100
 *  - 严格五言: 每句必须正好 5 个汉字，多/少 1 字 -15
 *  - 二四句押韵: 第 2、4 句尾字同韵部 +40；不同韵 -40
 *  - 首句入韵（第 1 句尾字与韵脚同韵部）: +10
 *  - 句内重字: 每处 -20
 *  - 泛滥字（单字全诗 >3 次）: 每超 1 次 -10
 *  - 有效诗句不足 4 句: 每少 1 句 -30
 */
class RhymeJudge(private val ctx: Context) {

    lateinit var rhymeOrder: List<String>            // 14 韵部名
    lateinit var rhymeChars: Map<String, String>     // 韵部 -> 全部字（字符串，查包含）
    private val charRhymes = HashMap<Char, MutableList<String>>()

    private val punct = setOf('，', '。', '！', '？', '；', '、')
    private val punctRegex = Regex("[，。！？；、]")

    fun load() {
        val j = JSONObject(ctx.assets.open("rhyme.json").bufferedReader(Charsets.UTF_8).use { it.readText() })
        rhymeOrder = j.getJSONArray("order").let { a -> (0 until a.length()).map { a.getString(it) } }
        rhymeChars = rhymeOrder.associateWith { r -> j.getJSONObject("chars").getString(r) }
        charRhymes.clear()
        for (r in rhymeOrder) for (c in rhymeChars[r]!!) {
            charRhymes.getOrPut(c) { ArrayList() }.add(r)
        }
    }

    fun rhymeOf(ch: Char): List<String> = charRhymes[ch] ?: emptyList()

    fun isRhyme(c1: Char, c2: Char): Boolean =
        (rhymeOf(c1).toSet() intersect rhymeOf(c2).toSet()).isNotEmpty()

    // ---------- 分行 ----------
    fun isTitleLine(line: String): Boolean = "《" in line || "》" in line

    private fun isAuthorLine(line: String): Boolean {
        val s = line.trim()
        if (s.isEmpty() || isTitleLine(s)) return false
        if (s.any { it in punct }) return false
        return s.length in 2..4
    }

    /** 供 PoemGenerator 判断作者行 */
    fun isAuthorLinePublic(line: String): Boolean = isAuthorLine(line)

    /** 句读标点集合（供 PoemGenerator 复用） */
    val linePunct: Set<Char> get() = punct

    fun splitPoemLines(text: String): List<String> {
        val poems = ArrayList<String>()
        for (raw in text.split("\n")) {
            val line = raw.trim()
            if (line.isEmpty() || isTitleLine(line) || isAuthorLine(line)) continue
            for (seg in line.split(punctRegex)) {
                val s = seg.trim()
                if (s.isNotEmpty()) poems.add(s)
            }
        }
        return poems
    }

    // ---------- 评分 ----------
    data class ScoreInfo(
        val score: Double,
        val lines: Int,
        val rhyme: String?,
        val secondFourOk: Boolean,
        val dup: Int,
        val overflow: Int,
        val short: Int,
        val titleInRhyme: Boolean,
    )

    fun scorePoem(text: String): ScoreInfo {
        val lines = splitPoemLines(text)
        var score = 100.0
        var rhyme: String? = null
        var secondFourOk = false
        var titleInRhyme = false
        var dup = 0; var overflow = 0; var short = 0

        for (ln in lines) if (ln.length != 5) { short++; score -= 15 }

        if (lines.size >= 4) {
            val t2 = lines[1].last(); val t4 = lines[3].last()
            val common = rhymeOf(t2).toSet() intersect rhymeOf(t4).toSet()
            if (common.isNotEmpty()) {
                secondFourOk = true
                rhyme = common.first()
                score += 40
                if (rhyme in rhymeOf(lines[0].last())) { titleInRhyme = true; score += 10 }
            } else {
                score -= 40
            }
        } else {
            score -= (4 - lines.size) * 30
        }

        for (ln in lines) {
            val seen = HashSet<Char>()
            for (c in ln) { if (c in seen) { dup++; score -= 20 }; seen.add(c) }
        }

        val cnt = HashMap<Char, Int>()
        for (ln in lines) for (c in ln) cnt[c] = (cnt[c] ?: 0) + 1
        for ((_, n) in cnt) if (n > 3) { overflow += n - 3; score -= 10.0 * (n - 3) }

        return ScoreInfo(score, lines.size, rhyme, secondFourOk, dup, overflow, short, titleInRhyme)
    }
}
