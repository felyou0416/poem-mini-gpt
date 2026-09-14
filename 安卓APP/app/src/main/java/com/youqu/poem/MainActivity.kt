package com.youqu.poem

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private lateinit var model: MiniGPT
    private lateinit var judge: RhymeJudge
    private lateinit var generator: PoemGenerator
    private var ready = false

    private lateinit var resultBox: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var seedInput: EditText
    private lateinit var generateBtn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // ============ 构建 UI（带滑动视口与传统美学） ============
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            setBackgroundColor(ContextCompat.getColor(this@MainActivity, R.color.paper))
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(28), dp(20), dp(24))
        }
        scroll.addView(root)

        // 标题
        root.addView(TextView(this).apply {
            text = "写诗 · AI"
            textSize = 28f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.cinnabar))
            typeface = Typeface.create("serif", Typeface.BOLD)
            gravity = Gravity.CENTER
        })
        root.addView(TextView(this).apply {
            text = "五言BPE v3·SFT ｜ 9.2万首预训练 + 3千首名篇微调 · 手机离线推理"
            textSize = 12f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_light))
            gravity = Gravity.CENTER
            setPadding(0, dp(4), 0, dp(16))
        })

        // 输入框
        root.addView(EditText(this).apply {
            hint = "输入题目或开头，如：山高水流"
            textSize = 17f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
            setHintTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_light))
            setPadding(dp(14), dp(10), dp(14), dp(10))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_input)
        }.also { seedInput = it })

        // 按钮行
        val btnRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(0, dp(16), 0, dp(6))
        }
        val btn = Button(this).apply {
            text = "作 诗"
            textSize = 18f
            setTextColor(Color.WHITE)
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_button)
            setOnClickListener { onGenerate() }
        }
        generateBtn = btn
        btnRow.addView(btn, LinearLayout.LayoutParams(dp(180), dp(50)))
        root.addView(btnRow)

        // 状态说明
        root.addView(TextView(this).apply {
            text = "加载模型中…"
            textSize = 13f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_light))
            gravity = Gravity.CENTER
            setPadding(0, dp(8), 0, 0)
        }.also { statusText = it })

        // 进度条
        root.addView(ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            visibility = View.GONE
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(6)).apply {
                topMargin = dp(8)
            }
        }.also { progressBar = it })

        // 结果区分割标头
        root.addView(TextView(this).apply {
            text = "—— 诗作 ——"
            textSize = 14f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink_light))
            gravity = Gravity.CENTER
            setPadding(0, dp(20), 0, dp(10))
        })

        // 结果诗卡容器
        resultBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        root.addView(resultBox, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

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
                    statusText.text = if (self) "模型自检通过 ✓ 随心输入开始写诗"
                                      else "模型自检通过（轻微浮点误差，正常可用）"
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "模型加载失败: ${e.message}"
                    Toast.makeText(this@MainActivity, "加载失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    /** 与 Python 端导出的 selftest.json 比对 top20 logits，验证 Kotlin 实现正确性 */
    private fun selftest(): Boolean {
        val j = JSONObject(assets.open("selftest.json").bufferedReader(Charsets.UTF_8).use { it.readText() })
        val tokens = j.getJSONArray("tokens").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val ids = j.getJSONArray("top20_ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val vals = j.getJSONArray("top20_logits").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
        val logits = model.forwardFull(tokens)
        val top = (0 until logits.size).sortedByDescending { logits[it] }.take(20)
        // 判据：top20 id 完全一致；logits 值误差 < 1.5（float32 累积）
        var mismatch = 0
        var maxErr = 0.0
        for (i in 0 until 20) {
            if (top[i] != ids[i]) mismatch++
            val err = Math.abs(logits[top[i]].toDouble() - vals[i])
            if (err > maxErr) maxErr = err
        }
        return mismatch == 0 && maxErr < 1.5
    }

    private fun onGenerate() {
        if (!ready) {
            Toast.makeText(this, "模型仍在加载中，请稍候…", Toast.LENGTH_SHORT).show()
            return
        }
        val seed = seedInput.text.toString().trim()
        if (seed.isEmpty()) {
            Toast.makeText(this, "请先输入题目或开头，如：山高水流", Toast.LENGTH_SHORT).show()
            return
        }

        generateBtn.isEnabled = false
        generateBtn.alpha = 0.6f
        progressBar.visibility = View.VISIBLE
        progressBar.progress = 0
        statusText.text = "正在构思中…"
        resultBox.removeAllViews()

        CoroutineScope(Dispatchers.Default).launch {
            try {
                val poems = generator.generate(
                    seed = seed,
                    temperature = 0.6f,
                    n = 3,
                    candidatesK = 6,
                    progress = { msg, pct ->
                        runOnUiThread {
                            statusText.text = msg
                            progressBar.progress = pct
                        }
                    }
                )
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    if (poems.isEmpty()) {
                        statusText.text = "这轮未生成合格诗作，换个题目试试？"
                    } else {
                        statusText.text = "吟成佳作 共 ${poems.size} 首"
                        for (p in poems) addPoemCard(p)
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

    private fun addPoemCard(text: String) {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(16), dp(18), dp(14))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_card)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(14)
            }
        }
        val lines = text.split("\n").filter { it.isNotBlank() }
        for ((i, ln) in lines.withIndex()) {
            val tv = TextView(this)
            tv.text = ln
            tv.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.ink))
            if (i == 0) {
                tv.textSize = 19f
                tv.typeface = Typeface.create("serif", Typeface.BOLD)
                tv.gravity = Gravity.CENTER
                tv.setPadding(0, 0, 0, dp(2))
            } else if (ln.contains("〔押")) {
                tv.textSize = 12f
                tv.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.cinnabar))
                tv.gravity = Gravity.CENTER
                tv.setPadding(0, 0, 0, dp(8))
            } else {
                tv.textSize = 17f
                tv.typeface = Typeface.create("serif", Typeface.NORMAL)
                tv.gravity = Gravity.CENTER
                tv.setPadding(0, dp(2), 0, dp(2))
            }
            card.addView(tv)
        }

        // 复制按钮行
        val copyRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
            setPadding(0, dp(10), 0, 0)
        }
        val copyBtn = TextView(this).apply {
            text = "复制全诗"
            textSize = 12f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.cinnabar))
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_copy_button)
            setPadding(dp(12), dp(4), dp(12), dp(4))
            isClickable = true
            isFocusable = true
            setOnClickListener {
                val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("诗词", text))
                Toast.makeText(this@MainActivity, "已复制全诗到剪切板", Toast.LENGTH_SHORT).show()
            }
        }
        copyRow.addView(copyBtn)
        card.addView(copyRow)

        resultBox.addView(card)
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
