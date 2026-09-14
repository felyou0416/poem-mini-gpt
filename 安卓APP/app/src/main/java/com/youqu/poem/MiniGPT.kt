package com.youqu.poem

import android.content.Context
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * MiniGPT_v3 纯 Kotlin 推理引擎（与 train_v3.py / poem.py 逐算子对齐）
 *
 * 结构: 6 层 Transformer, 192 维, 6 头(32/头), RoPE(半拆分), RMSNorm, GELU(精确 erf),
 *       权重共享 embedding/lm_head, 因果掩码。
 * 推理: 增量 KV-cache（每步只算新 token 行），保证手机端速度。
 */
class MiniGPT(private val ctx: Context) {

    data class Weight(val offset: Int, val shape: IntArray)

    // ---- 超参 ----
    lateinit var meta: JSONObject
    var nLayer = 6; var nEmb = 192; var nHead = 6; var headDim = 32
    var blockSize = 128; var vocabSize = 4006
    var normEps = 1e-6f; var ropeBase = 10000.0

    // ---- 词表 ----
    lateinit var itos: Array<String>      // id -> 字/特殊token
    lateinit var stoi: Map<String, Int>

    // ---- 特殊 token ----
    lateinit var special: Map<String, Int>

    // ---- 权重 ----
    lateinit var weights: FloatArray      // 全量 float32
    lateinit var manifest: Map<String, Weight>
    private val wcache = HashMap<String, FloatArray>()   // key -> 切片引用（每步复用，不拷贝）

    // ---- RoPE 预计算 ----
    private var cosT: Array<FloatArray> = emptyArray()   // [blockSize, headDim/2]
    private var sinT: Array<FloatArray> = emptyArray()

    fun load() {
        meta = JSONObject(readAsset("meta.json"))
        nLayer = meta.getInt("n_layer"); nEmb = meta.getInt("n_embd")
        nHead = meta.getInt("n_head"); blockSize = meta.getInt("block_size")
        vocabSize = meta.getInt("vocab_size"); headDim = nEmb / nHead
        normEps = meta.optDouble("norm_eps", 1e-6).toFloat()
        ropeBase = meta.optDouble("rope_base", 10000.0)
        special = JSONObject(meta.getJSONObject("special").toString()).let { j ->
            j.keys().asSequence().associateWith { j.getInt(it) }
        }

        // 词表: {"0":"<|bos|>", ...}
        val vj = JSONObject(readAsset("vocab.json"))
        itos = Array(vocabSize) { i -> vj.optString(i.toString(), "□") }
        stoi = itos.indices.associate { itos[it] to it }

        // manifest
        val mj = JSONObject(readAsset("manifest.json"))
        manifest = mj.keys().asSequence().associate { k ->
            val o = mj.getJSONObject(k)
            k to Weight(o.getInt("offset"), o.getJSONArray("shape").let { a ->
                IntArray(a.length()) { a.getInt(it) } })
        }

        // model.bin
        val raw = readAssetBytes("model.bin")
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        weights = FloatArray(bb.remaining())
        bb.get(weights)

        // RoPE 表
        val half = headDim / 2
        val inv = FloatArray(half) { (1.0 / Math.pow(ropeBase, (it * 2).toDouble() / headDim)).toFloat() }
        cosT = Array(blockSize) { t -> FloatArray(half) { i -> Math.cos(t * inv[i].toDouble()).toFloat() } }
        sinT = Array(blockSize) { t -> FloatArray(half) { i -> Math.sin(t * inv[i].toDouble()).toFloat() } }
    }

    // ================= 权重访问 =================
    fun weight(key: String): FloatArray {
        wcache[key]?.let { return it }
        val w = manifest[key] ?: throw IllegalStateException("权重缺失: $key")
        val n = w.shape[0] * (if (w.shape.size > 1) w.shape[1] else 1)
        val arr = FloatArray(n) { weights[w.offset + it] }
        wcache[key] = arr
        return arr
    }

    /** 行式矩阵访问视图（避免整矩阵拷贝）：取 [dim] 行向量 */
    private fun rowAt(w: FloatArray, cols: Int, r: Int): FloatArray {
        val out = FloatArray(cols)
        System.arraycopy(w, r * cols, out, 0, cols)
        return out
    }

    // ================= KV 缓存 =================
    class KvCache(val nLayer: Int, val nHead: Int, val headDim: Int) {
        // 每层每头一个动态列表：k[l][h] = List<FloatArray(headDim)>
        val keys = Array(nLayer) { Array(nHead) { ArrayList<FloatArray>() } }
        val vals = Array(nLayer) { Array(nHead) { ArrayList<FloatArray>() } }
        fun clear() {
            for (l in 0 until nLayer) for (h in 0 until nHead) { keys[l][h].clear(); vals[l][h].clear() }
        }
    }

    // ================= 算子 =================
    private fun rmsNorm(x: FloatArray, w: FloatArray): FloatArray {
        var s = 0f
        for (v in x) s += v * v
        val rms = Math.sqrt((s / x.size + normEps).toDouble()).toFloat()
        val out = FloatArray(x.size)
        for (i in x.indices) out[i] = x[i] / rms * w[i]
        return out
    }

    private fun matVec(m: FloatArray, cols: Int, x: FloatArray, bias: FloatArray?): FloatArray {
        val rows = m.size / cols
        val out = FloatArray(rows)
        for (r in 0 until rows) {
            var acc = bias?.get(r) ?: 0f
            val base = r * cols
            for (c in 0 until cols) acc += m[base + c] * x[c]
            out[r] = acc
        }
        return out
    }

    private fun gelu(x: Float): Float {
        val v = x.toDouble()
        val e = erf(v / Math.sqrt(2.0))
        return (0.5 * v * (1.0 + e)).toFloat()
    }

    private fun erf(x: Double): Double {
        val sign = if (x < 0) -1.0 else 1.0
        val ax = Math.abs(x)
        val t = 1.0 / (1.0 + 0.3275911 * ax)
        val y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t *
            Math.exp(-ax * ax)
        return sign * y
    }

    // ================= 单步推理（增量，KV-cache） =================
    /**
     * 输入当前最后一个 token id 与绝对位置 pos，返回 [vocabSize] logits，
     * 并把该步 k/v 写入 cache（需在调用后立刻使用 logits，cache 已被推进）。
     */
    fun forwardStep(tok: Int, pos: Int, cache: KvCache): FloatArray {
        val x0 = rowAt(weight("tok_emb.weight"), nEmb, tok)
        var x = x0

        for (l in 0 until nLayer) {
            // ---- Attention 分支 ----
            val n1 = weight("blocks.$l.norm1.weight")
            val xn = rmsNorm(x, n1)

            val cAttnW = weight("blocks.$l.attn.c_attn.weight")   // [3*E, E]
            val cAttnB = weight("blocks.$l.attn.c_attn.bias")     // [3*E]
            val qkv = matVec(cAttnW, nEmb, xn, cAttnB)            // [3E]
            val q = qkv.copyOfRange(0, nEmb)
            val k = qkv.copyOfRange(nEmb, 2 * nEmb)
            val v = qkv.copyOfRange(2 * nEmb, 3 * nEmb)

            // RoPE（half-split）
            val d = headDim / 2
            val cos = cosT[pos]; val sin = sinT[pos]
            for (h in 0 until nHead) {
                val qh = FloatArray(headDim); val kh = FloatArray(headDim)
                for (i in 0 until headDim) { qh[i] = q[h * headDim + i]; kh[i] = k[h * headDim + i] }
                for (i in 0 until d) {
                    val q1 = qh[i]; val q2 = qh[i + d]
                    qh[i] = q1 * cos[i] - q2 * sin[i]; qh[i + d] = q2 * cos[i] + q1 * sin[i]
                    val k1 = kh[i]; val k2 = kh[i + d]
                    kh[i] = k1 * cos[i] - k2 * sin[i]; kh[i + d] = k2 * cos[i] + k1 * sin[i]
                }
                System.arraycopy(qh, 0, q, h * headDim, headDim)
                System.arraycopy(kh, 0, k, h * headDim, headDim)
            }

            val scale = (1.0 / Math.sqrt(headDim.toDouble())).toFloat()
            val attnOut = FloatArray(nEmb)
            for (h in 0 until nHead) {
                val kh = cache.keys[l][h]; val vh = cache.vals[l][h]
                val qh = FloatArray(headDim); System.arraycopy(q, h * headDim, qh, 0, headDim)
                val kv = FloatArray(headDim); System.arraycopy(k, h * headDim, kv, 0, headDim)
                val vv = FloatArray(headDim); System.arraycopy(v, h * headDim, vv, 0, headDim)
                kh.add(kv); vh.add(vv)   // 先入库再算注意力（本步可见自身）

                val n = kh.size
                val scores = FloatArray(n)
                var maxS = Float.NEGATIVE_INFINITY
                for (j in 0 until n) {
                    val kj = kh[j]
                    var s = 0f
                    for (i in 0 until headDim) s += qh[i] * kj[i]
                    s *= scale
                    scores[j] = s
                    if (s > maxS) maxS = s
                }
                var sum = 0f
                for (j in 0 until n) { scores[j] = Math.exp((scores[j] - maxS).toDouble()).toFloat(); sum += scores[j] }
                for (h2 in 0 until headDim) {
                    var acc = 0f
                    for (j in 0 until n) acc += scores[j] / sum * vh[j][h2]
                    attnOut[h * headDim + h2] = acc
                }
            }

            val cProjW = weight("blocks.$l.attn.c_proj.weight")   // [E,E]
            val cProjB = weight("blocks.$l.attn.c_proj.bias")
            val attnProj = matVec(cProjW, nEmb, attnOut, cProjB)
            for (i in 0 until nEmb) x[i] += attnProj[i]

            // ---- MLP 分支 ----
            val n2 = weight("blocks.$l.norm2.weight")
            val xn2 = rmsNorm(x, n2)
            val fcW = weight("blocks.$l.mlp.fc.weight")           // [4E, E]
            val fcB = weight("blocks.$l.mlp.fc.bias")             // [4E]
            val hidden = matVec(fcW, nEmb, xn2, fcB)
            for (i in hidden.indices) hidden[i] = gelu(hidden[i])
            val projW = weight("blocks.$l.mlp.proj.weight")       // [E, 4E]
            val projB = weight("blocks.$l.mlp.proj.bias")
            val mlpOut = matVec(projW, hidden.size, hidden, projB)
            for (i in 0 until nEmb) x[i] += mlpOut[i]
        }

        // ---- final norm + lm_head（权重共享） ----
        val nf = weight("norm_f.weight")
        val xf = rmsNorm(x, nf)
        val tokEmb = weight("tok_emb.weight")
        val logits = FloatArray(vocabSize)
        for (i in 0 until vocabSize) {
            var s = 0f
            val base = i * nEmb
            for (d in 0 until nEmb) s += xf[d] * tokEmb[base + d]
            logits[i] = s
        }
        return logits
    }

    /** 全序列前向（用于启动自检 selftest），返回序列末尾 logits */
    fun forwardFull(tokens: IntArray): FloatArray {
        val cache = KvCache(nLayer, nHead, headDim)
        var logits = FloatArray(vocabSize)
        for (pos in tokens.indices) {
            logits = forwardStep(tokens[pos], pos, cache)
        }
        return logits
    }

    // ================= 采样 =================
    /** 复刻 poem.py sample_next：logits/温度 → top_k 截断 → softmax → 多项式采样 */
    fun sample(logits: FloatArray, temperature: Float, topK: Int, rnd: java.util.Random): Int {
        val temp = Math.max(temperature, 1e-6f)
        val scaled = FloatArray(logits.size) { logits[it] / temp }
        // top-k
        val k = Math.min(topK, scaled.size)
        val top = scaled.indices.sortedByDescending { scaled[it] }.take(k)
        val th = scaled[top.last()]
        for (i in scaled.indices) if (scaled[i] < th) scaled[i] = Float.NEGATIVE_INFINITY
        var anyFinite = false
        for (s in scaled) if (s.isFinite()) { anyFinite = true; break }
        if (!anyFinite) scaled.fill(0f)
        // softmax
        var maxS = Float.NEGATIVE_INFINITY
        for (s in scaled) if (s.isFinite() && s > maxS) maxS = s
        var sum = 0.0
        for (i in scaled.indices) {
            scaled[i] = if (scaled[i].isFinite()) Math.exp((scaled[i] - maxS).toDouble()).toFloat() else 0f
            sum += scaled[i]
        }
        var r = rnd.nextDouble() * sum
        for (i in scaled.indices) {
            r -= scaled[i]
            if (r <= 0) return i
        }
        return scaled.indices.last { scaled[it] > 0f }
    }

    private fun readAsset(name: String): String =
        ctx.assets.open(name).bufferedReader(Charsets.UTF_8).use { it.readText() }

    private fun readAssetBytes(name: String): ByteArray =
        ctx.assets.open(name).use { it.readBytes() }
}
