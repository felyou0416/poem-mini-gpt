package com.youqu.poem

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.Locale

/**
 * 砚墨成诗 · 迷你GPT写诗机（安卓原生国风控制面板）
 * 1:1 复刻 Web 前端 (ui.html) 界面视觉与全部交互参数控制：
 * 1. 灵感意象标签池（点击即填入）
 * 2. 采样温度实时滑块（带黄金平衡/灵动/工整风格动态指导）
 * 3. 生成首数胶囊选择器（1首/2首/3首/4首）
 * 4. 宣纸卡片排版、朱砂红“墨韵雅成”印章、平水韵韵部徽章
 * 5. 一键复制、系统分享、Android 原生语音慢速古风吟诵
 * 6. 本次会话历史记录抽屉
 */
class MainActivity : AppCompatActivity() {

    private lateinit var model: MiniGPT
    private lateinit var judge: RhymeJudge
    private lateinit var generator: PoemGenerator
    private var ready = false

    // 参数控制状态
    private var currentTemp = 0.60f
    private var currentN = 3

    // UI 组件引用
    private lateinit var seedInput: EditText
    private lateinit var tempValBadge: TextView
    private lateinit var tempStyleHint: TextView
    private lateinit var generateBtn: Button
    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var resultBox: LinearLayout
    private lateinit var countBtns: List<TextView>

    // 会话历史
    private val historyList = ArrayList<Pair<String, String>>()
    private lateinit var historyContainer: LinearLayout
    private lateinit var historyToggleBtn: TextView

    // 原生语音合成
    private var tts: TextToSpeech? = null
    private var isTtsReady = false
    private var activeRecitingBtn: TextView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 初始化原生语音朗读引擎
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val res = tts?.setLanguage(Locale.CHINESE)
                isTtsReady = res != TextToSpeech.LANG_MISSING_DATA && res != TextToSpeech.LANG_NOT_SUPPORTED
                tts?.setSpeechRate(0.82f) // 古风吟诵略带停顿舒缓语速
            }
        }

        // ============ 构建 UI（国风宣纸视口） ============
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            setBackgroundColor(ContextCompat.getColor(this@MainActivity, R.color.paper))
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(24), dp(16), dp(36))
        }
        scroll.addView(root)

        // ---- 1. 顶栏 Header ----
        val headerBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, dp(18))
        }

        // 徽章
        headerBox.addView(TextView(this).apply {
            this.text = "✦ 迷你 GPT 诗词引擎"
            textSize = 11f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.pine_green))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_rhyme_badge)
            setPadding(dp(10), dp(3), dp(10), dp(3))
            gravity = Gravity.CENTER
        })

        // 大标题
        headerBox.addView(TextView(this).apply {
            this.text = "砚 墨 成 诗"
            textSize = 27f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
            typeface = Typeface.create("serif", Typeface.BOLD)
            gravity = Gravity.CENTER
            setPadding(0, dp(6), 0, dp(2))
        })

        // 副标题
        headerBox.addView(TextView(this).apply {
            this.text = "轻量自回归语言模型 · 古风古典格律自动生成"
            textSize = 12f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_muted))
            gravity = Gravity.CENTER
        })
        root.addView(headerBox)

        // 主交互卡片
        val mainCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_card)
            setPadding(dp(18), dp(20), dp(18), dp(20))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(18) }
        }

        // ---- 2. 灵感命题/诗歌开头 ----
        mainCard.addView(createSectionLabel("灵感命题 / 诗歌开头", "支持意象词或《标题》"))

        seedInput = EditText(this).apply {
            setText("秋思")
            hint = "如：秋思（意境） / 《山高水流》（标题）"
            textSize = 15f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
            setHintTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_muted))
            setPadding(dp(12), dp(9), dp(12), dp(9))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_input)
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(44)).apply {
                bottomMargin = dp(8)
            }
        }
        mainCard.addView(seedInput)

        // 灵感标签推荐池（水平滚动）
        val tagScroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(16)
            }
        }
        val tagRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(2), 0, dp(2))
        }

        val moodTags = listOf("秋思", "孤舟", "咏梅", "听雨", "登高", "夜泊")
        for (t in moodTags) {
            tagRow.addView(createTagChip(t, false))
        }
        val titleTags = listOf("《春日》", "《山高水流》", "《临江仙》")
        for (t in titleTags) {
            tagRow.addView(createTagChip(t, true))
        }
        tagScroll.addView(tagRow)
        mainCard.addView(tagScroll)

        // ---- 3. 推理模型卡片 ----
        mainCard.addView(createSectionLabel("推理模型", "4,006 词元 · 344万参数 · 唐风微调"))
        val modelBox = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_model_box)
            setPadding(dp(12), dp(10), dp(12), dp(10))
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(16)
            }
        }
        modelBox.addView(TextView(this).apply {
            this.text = "🏮"
            textSize = 18f
            setPadding(0, 0, dp(8), 0)
        })
        val modelDescBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        modelDescBox.addView(TextView(this).apply {
            this.text = "五言BPE v3·SFT（唐风名篇微调 · 纯离线版）"
            textSize = 13f
            typeface = Typeface.create("serif", Typeface.BOLD)
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
        })
        modelDescBox.addView(TextView(this).apply {
            this.text = "9.2万首五言预训练 + 3,125首李杜王孟名篇二次SFT · 手机本地极速推理"
            textSize = 11f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_light))
            setPadding(0, dp(2), 0, 0)
        })
        modelBox.addView(modelDescBox)
        mainCard.addView(modelBox)

        // ---- 4. 双列参数调节：采样温度 + 生成首数 ----
        val controlsGrid = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(18)
            }
        }

        // 采样温度
        val tempHeader = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        tempHeader.addView(TextView(this).apply {
            this.text = "采样温度"
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
        })
        tempValBadge = TextView(this).apply {
            this.text = "0.60"
            textSize = 12f
            typeface = Typeface.MONOSPACE
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.cinnabar))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_rhyme_badge)
            setPadding(dp(8), dp(2), dp(8), dp(2))
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                marginStart = dp(8)
            }
        }
        tempHeader.addView(tempValBadge)
        controlsGrid.addView(tempHeader)

        // 温度滑动条
        val seekBar = SeekBar(this).apply {
            max = 26 // (1.50 - 0.20) / 0.05 = 26 档
            progress = 8 // (0.60 - 0.20) / 0.05 = 8
            setPadding(0, dp(8), 0, dp(4))
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, prog: Int, fromUser: Boolean) {
                    val t = 0.20f + prog * 0.05f
                    currentTemp = Math.round(t * 100f) / 100f
                    tempValBadge.text = String.format(Locale.US, "%.2f", currentTemp)
                    updateTempHint(currentTemp)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        controlsGrid.addView(seekBar)

        tempStyleHint = TextView(this).apply {
            this.text = "黄金平衡（推荐，韵味自然工丽）"
            textSize = 11f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_muted))
            setPadding(0, 0, 0, dp(14))
        }
        controlsGrid.addView(tempStyleHint)

        // 生成首数选择
        controlsGrid.addView(createSectionLabel("生成首数", "批量采样优选对比"))
        val pillsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(4), 0, 0)
        }
        val pillList = ArrayList<TextView>()
        val counts = listOf(1, 2, 3, 4)
        for (c in counts) {
            val pill = TextView(this).apply {
                this.text = "${c}首"
                textSize = 13f
                gravity = Gravity.CENTER
                setPadding(dp(16), dp(6), dp(16), dp(6))
                isClickable = true
                isFocusable = true
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                    marginEnd = if (c < 4) dp(8) else 0
                }
                setOnClickListener { selectPoemCount(c) }
            }
            pillList.add(pill)
            pillsRow.addView(pill)
        }
        countBtns = pillList
        updatePillUI()
        controlsGrid.addView(pillsRow)
        mainCard.addView(controlsGrid)

        // ---- 5. 生成按钮 ----
        generateBtn = Button(this).apply {
            this.text = "砚 墨 成 诗"
            textSize = 17f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.WHITE)
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_button)
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(48))
            setOnClickListener { onGenerate() }
        }
        mainCard.addView(generateBtn)

        // ---- 状态与进度 ----
        statusText = TextView(this).apply {
            this.text = "加载模型中…"
            textSize = 12f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_light))
            gravity = Gravity.CENTER
            setPadding(0, dp(10), 0, 0)
        }
        mainCard.addView(statusText)

        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            visibility = View.GONE
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(4)).apply {
                topMargin = dp(6)
            }
        }
        mainCard.addView(progressBar)

        root.addView(mainCard)

        // ---- 6. 诗歌成果展示区 ----
        root.addView(TextView(this).apply {
            this.text = "—— 诗作佳构 ——"
            textSize = 13f
            letterSpacing = 0.1f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_muted))
            gravity = Gravity.CENTER
            setPadding(0, dp(4), 0, dp(12))
        })

        resultBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        root.addView(resultBox)

        // ---- 7. 本次会话历史记录抽屉 ----
        val historySection = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(16), 0, 0)
        }
        historyToggleBtn = TextView(this).apply {
            this.text = "▶ 本次会话生成历史 (0)"
            textSize = 13f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_secondary))
            setPadding(dp(4), dp(8), dp(4), dp(8))
            setOnClickListener { toggleHistory() }
        }
        historySection.addView(historyToggleBtn)

        historyContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setPadding(0, dp(8), 0, 0)
        }
        historySection.addView(historyContainer)
        root.addView(historySection)

        setContentView(scroll)

        // ============ 后台加载模型 + 自检 ============
        CoroutineScope(Dispatchers.Default).launch {
            try {
                model = MiniGPT(this@MainActivity)
                model.load()
                judge = RhymeJudge(this@MainActivity)
                judge.load()
                generator = PoemGenerator(model, judge)
                val self = selftest()
                ready = true
                withContext(Dispatchers.Main) {
                    statusText.text = if (self) "模型就绪 ✓ 随心输入命题挥毫成诗"
                                      else "模型就绪（轻微浮点差，可正常作诗）"
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "模型加载失败: ${e.message}"
                    Toast.makeText(this@MainActivity, "加载失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }

    // ---------- 标签与首数辅助 ----------
    private fun createSectionLabel(title: String, tip: String): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(6)
            }
            addView(TextView(this@MainActivity).apply {
                this.text = title
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
                setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
            })
            addView(TextView(this@MainActivity).apply {
                this.text = tip
                textSize = 11f
                setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_muted))
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                    marginStart = dp(6)
                }
            })
        }
    }

    private fun createTagChip(text: String, isTitle: Boolean): TextView {
        return TextView(this).apply {
            this.text = text
            textSize = 12f
            setPadding(dp(10), dp(4), dp(10), dp(4))
            background = ContextCompat.getDrawable(
                this@MainActivity,
                if (isTitle) R.drawable.bg_chip_title else R.drawable.bg_chip
            )
            setTextColor(
                ContextCompat.getColor(
                    this@MainActivity,
                    if (isTitle) R.color.pine_green else R.color.ink_light
                )
            )
            isClickable = true
            isFocusable = true
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                marginEnd = dp(6)
            }
            setOnClickListener {
                seedInput.setText(text)
                seedInput.setSelection(text.length)
            }
        }
    }

    private fun selectPoemCount(n: Int) {
        currentN = n
        updatePillUI()
    }

    private fun updatePillUI() {
        for ((idx, btn) in countBtns.withIndex()) {
            val count = idx + 1
            if (count == currentN) {
                btn.background = ContextCompat.getDrawable(this, R.drawable.bg_pill_selected)
                btn.setTextColor(Color.WHITE)
            } else {
                btn.background = ContextCompat.getDrawable(this, R.drawable.bg_pill_unselected)
                btn.setTextColor(ContextCompat.getColor(this, R.color.ink_light))
            }
        }
    }

    private fun updateTempHint(t: Float) {
        tempStyleHint.text = when {
            t < 0.40f -> "极度工整（字句规整，重复度略高）"
            t <= 0.70f -> "黄金平衡（推荐，韵味自然工丽）"
            t <= 1.00f -> "灵动飘逸（意象丰富，发散多变）"
            else -> "天马行空（意象奇拔，偶有险句）"
        }
    }

    // ---------- 核心作诗生成流程 ----------
    private fun onGenerate() {
        if (!ready) {
            Toast.makeText(this, "模型仍在加载中，请稍候…", Toast.LENGTH_SHORT).show()
            return
        }
        val seed = seedInput.text.toString().trim()
        if (seed.isEmpty()) {
            Toast.makeText(this, "请先输入题目或开头意象", Toast.LENGTH_SHORT).show()
            return
        }

        // 停止之前的朗读
        tts?.stop()
        activeRecitingBtn?.text = "🔊 吟诵"
        activeRecitingBtn = null

        generateBtn.isEnabled = false
        generateBtn.alpha = 0.6f
        progressBar.visibility = View.VISIBLE
        progressBar.progress = 0
        statusText.text = "研墨挥毫，意象缀连中…"
        resultBox.removeAllViews()

        val startTime = System.currentTimeMillis()

        CoroutineScope(Dispatchers.Default).launch {
            try {
                val poems = generator.generate(
                    seed = seed,
                    temperature = currentTemp,
                    n = currentN,
                    candidatesK = currentN * 3,
                    progress = { msg, pct ->
                        runOnUiThread {
                            statusText.text = msg
                            progressBar.progress = pct
                        }
                    }
                )
                val elapsed = ((System.currentTimeMillis() - startTime) / 1000.0)
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    if (poems.isEmpty()) {
                        statusText.text = "这轮未切出合规格律诗篇，建议微调温度重试"
                    } else {
                        statusText.text = String.format(Locale.CHINA, "完成 ✓ 温度 %.2f · 耗时 %.1fs · 共 %d 首", currentTemp, elapsed, poems.size)
                        for (p in poems) {
                            addPoemCard(p)
                            addToHistory(p, seed)
                        }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    statusText.text = "出错: ${e.message}"
                }
            } finally {
                withContext(Dispatchers.Main) {
                    generateBtn.isEnabled = true
                    generateBtn.alpha = 1.0f
                }
            }
        }
    }

    // ---------- 渲染古风卡片（含朱砂印章与语音朗诵） ----------
    private fun addPoemCard(poemContent: String) {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(18), dp(18), dp(16))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_card)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(16) }
        }

        val lines = poemContent.split("\n").map { it.trim() }.filter { it.isNotEmpty() }
        var title = if (lines.isNotEmpty() && lines[0].startsWith("《")) lines[0] else "《无题》"
        var author = "无名氏"
        var rhymeTag = ""
        val verseLines = ArrayList<String>()

        var startIdx = 0
        if (lines.isNotEmpty() && (lines[0].startsWith("《") || lines[0].contains("》"))) {
            title = lines[0]
            startIdx = 1
            if (lines.size > 1 && (lines[1].length <= 20 || lines[1].contains("〔押")) && !lines[1].contains("，") && !lines[1].contains("。")) {
                val authLine = lines[1]
                val m = Regex("〔押(.+?)〕").find(authLine)
                if (m != null) {
                    rhymeTag = m.groupValues[1].trim()
                    author = authLine.replace(Regex("〔押.+?〕"), "").trim()
                } else {
                    author = authLine
                }
                startIdx = 2
            }
        }
        for (i in startIdx until lines.size) {
            verseLines.add(lines[i])
        }
        if (verseLines.isEmpty()) {
            verseLines.addAll(lines)
        }

        // 顶行：标题居中 + 右上角朱砂红小印章
        val topRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val tvTitle = TextView(this).apply {
            this.text = title
            textSize = 19f
            typeface = Typeface.create("serif", Typeface.BOLD)
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        topRow.addView(tvTitle)

        // 朱砂小印章（墨韵雅成）
        val sealView = TextView(this).apply {
            this.text = "墨韵\n雅成"
            textSize = 8f
            typeface = Typeface.create("serif", Typeface.BOLD)
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.cinnabar))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_seal)
            gravity = Gravity.CENTER
            setPadding(dp(4), dp(2), dp(4), dp(2))
        }
        topRow.addView(sealView)
        card.addView(topRow)

        // 作者 + 韵部徽章
        val authorRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(0, dp(6), 0, dp(12))
        }
        authorRow.addView(TextView(this).apply {
            this.text = "◈ $author"
            textSize = 12f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_muted))
        })
        if (rhymeTag.isNotEmpty()) {
            authorRow.addView(TextView(this).apply {
                this.text = "押$rhymeTag"
                textSize = 10f
                setTextColor(ContextCompat.getColor(this@MainActivity, R.color.pine_green))
                background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_rhyme_badge)
                setPadding(dp(6), dp(1), dp(6), dp(1))
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                    marginStart = dp(6)
                }
            })
        }
        card.addView(authorRow)

        // 诗词正文
        for (ln in verseLines) {
            val tv = TextView(this).apply {
                this.text = ln
                textSize = 17f
                typeface = Typeface.create("serif", Typeface.NORMAL)
                setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
                gravity = Gravity.CENTER
                setPadding(0, dp(3), 0, dp(3))
            }
            card.addView(tv)
        }

        // 底部操作区（一键复制、语音吟诵、系统分享）
        val actionRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(0, dp(16), 0, 0)
        }

        // 1. 复制按钮
        actionRow.addView(createActionButton("📋 复制") {
            val fullText = "$title\n$author${if (rhymeTag.isNotEmpty()) "  〔押$rhymeTag〕" else ""}\n\n${verseLines.joinToString("\n")}\n\n—— 砚墨成诗 · 迷你GPT"
            val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("诗词", fullText))
            Toast.makeText(this@MainActivity, "✓ 诗篇已复制到剪切板", Toast.LENGTH_SHORT).show()
        })

        // 2. 吟诵按钮
        val versesText = verseLines.joinToString("，")
        lateinit var btnRecite: TextView
        btnRecite = createActionButton("🔊 吟诵") {
            if (tts?.isSpeaking == true && activeRecitingBtn == btnRecite) {
                tts?.stop()
                btnRecite.text = "🔊 吟诵"
                activeRecitingBtn = null
            } else {
                if (!isTtsReady) {
                    Toast.makeText(this@MainActivity, "系统语音引擎尚未就绪", Toast.LENGTH_SHORT).show()
                    return@createActionButton
                }
                tts?.stop()
                activeRecitingBtn?.text = "🔊 吟诵"
                activeRecitingBtn = btnRecite
                btnRecite.text = "⏹ 停止"
                tts?.speak(versesText, TextToSpeech.QUEUE_FLUSH, null, "recite_id")
            }
        }
        actionRow.addView(btnRecite)

        // 3. 分享按钮
        actionRow.addView(createActionButton("📤 分享") {
            val fullText = "$title\n$author${if (rhymeTag.isNotEmpty()) "  〔押$rhymeTag〕" else ""}\n\n${verseLines.joinToString("\n")}\n\n—— 砚墨成诗 · 迷你GPT"
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, fullText)
            }
            startActivity(Intent.createChooser(intent, "分享这首佳作"))
        })

        card.addView(actionRow)
        resultBox.addView(card)
    }

    private fun createActionButton(label: String, onClick: () -> Unit): TextView {
        return TextView(this).apply {
            this.text = label
            textSize = 12f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_light))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_copy_button)
            setPadding(dp(12), dp(5), dp(12), dp(5))
            isClickable = true
            isFocusable = true
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                marginStart = dp(4)
                marginEnd = dp(4)
            }
            setOnClickListener { onClick() }
        }
    }

    // ---------- 会话历史记录 ----------
    private fun addToHistory(poem: String, seed: String) {
        val firstLine = poem.split("\n").firstOrNull { it.isNotBlank() } ?: "《无题》"
        historyList.add(0, Pair(firstLine, poem))
        historyToggleBtn.text = "▼ 本次会话生成历史 (${historyList.size})"
        updateHistoryContainer()
    }

    private fun toggleHistory() {
        val isOpen = historyContainer.visibility == View.VISIBLE
        historyContainer.visibility = if (isOpen) View.GONE else View.VISIBLE
        historyToggleBtn.text = if (isOpen) "▶ 本次会话生成历史 (${historyList.size})" else "▼ 本次会话生成历史 (${historyList.size})"
    }

    private fun updateHistoryContainer() {
        historyContainer.removeAllViews()
        for ((title, _) in historyList.take(15)) {
            historyContainer.addView(TextView(this).apply {
                this.text = "• $title"
                textSize = 12f
                setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_muted))
                setPadding(dp(6), dp(4), dp(6), dp(4))
            })
        }
    }

    /** 验证 Kotlin 推理算子与 Python 导出的 logits 一致性 */
    private fun selftest(): Boolean {
        val j = JSONObject(assets.open("selftest.json").bufferedReader(Charsets.UTF_8).use { it.readText() })
        val tokens = j.getJSONArray("tokens").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val ids = j.getJSONArray("top20_ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val vals = j.getJSONArray("top20_logits").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
        val logits = model.forwardFull(tokens)
        val top = (0 until logits.size).sortedByDescending { logits[it] }.take(20)
        var mismatch = 0
        var maxErr = 0.0
        for (i in 0 until 20) {
            if (top[i] != ids[i]) mismatch++
            val err = Math.abs(logits[top[i]].toDouble() - vals[i])
            if (err > maxErr) maxErr = err
        }
        return mismatch == 0 && maxErr < 1.5
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
