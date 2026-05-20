package com.audioclassifier
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtException
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import org.merlyn.kotlinspeechfeatures.SpeechFeatures
import java.nio.FloatBuffer
import java.io.File
import kotlin.math.exp



class MainActivity : AppCompatActivity() {

    private lateinit var ortEnv: OrtEnvironment  // ONNX Runtime 环境
    private lateinit var ortSession: OrtSession  // ONNX Runtime 会话
    private val speechFeatures = SpeechFeatures()

    val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化 ONNX Runtime 环境和会话
        initOnnxModel()

        val filePath = "0d6d7360_nohash_4.wav"

        val wav = loadWavFile(readAudioFileFromAsset(filePath))
        val floatArray = wav.map { it.toFloat() }.toFloatArray()
        val result = speechFeatures.mfcc(floatArray, 16000, nFilt = 64, numCep = 13)
        println("mfcc feature: [${result.size}, ${result[0].size}]") // 打印 mfcc 的形状

        val inferResult = processFeatures(result)
        val onnxResult = runOnnxInference(inferResult)
        Log.d(TAG, "onnxResult $onnxResult")

    }



    private fun loadWavFile(file: File): IntArray {
        val wavFile = WavFile.openWavFile(file)
        val numFrames = wavFile.numFrames.toInt()
        val channels = wavFile.numChannels
        val loopCounter: Int = (numFrames * channels / 4096) + 1

        val intBuffer = IntArray(numFrames)

        // 读取音频数据
        for (i in 0 until loopCounter) {
            val shortBuffer = ShortArray(4096)  // 假设读取的是16位数据
            val framesRead = wavFile.readFrames(shortBuffer, 4096) // 读取音频帧

            // 将读取的 short 数据转换成 int 数据
            for (j in 0 until framesRead) {
                // 将 short 类型数据转换为 int，并处理符号
                intBuffer[i * 4096 + j] = shortBuffer[j].toInt()
            }
        }

        return intBuffer
    }



    private fun readAudioFileFromAsset(name: String): File {
        val context = getApplication()
        val cacheDir = context.cacheDir
        return File("$cacheDir/$name").apply { writeBytes(context.assets.open("$name").readBytes()) }
    }


    // 初始化 ONNX 模型
    private fun initOnnxModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions()

            // 从 assets 文件夹中加载 ONNX 模型
            val modelBytes = readAssetModel("best_model.onnx")

            ortSession = ortEnv.createSession(modelBytes, sessionOptions)

            Log.d("ONNX", "ONNX model loaded successfully.")
        } catch (e: OrtException) {
            Log.e("ONNX", "Error loading ONNX model", e)
        }
    }



    private fun processFeatures(wavFeature: Array<FloatArray>): Array<Array<FloatArray>> {
        val padLen = 128
        val numRows = wavFeature.size // 即 x
        val numCols = wavFeature[0].size // 即 13

        // Step 1: 创建一个包含三个相同数据的数组
        val feature = Array(3) { Array(numRows) { FloatArray(numCols) } }
        for (i in 0 until 3) {
            for (j in 0 until numRows) {
                for (k in 0 until numCols) {
                    feature[i][j][k] = wavFeature[j][k]
                }
            }
        }

        // Step 2: 填充或截取数据
        val paddedFeature = Array(3) { Array(padLen) { FloatArray(numCols) } }
        for (i in 0 until 3) {
            for (h in 0 until padLen) {
                for (w in 0 until numCols) {
                    paddedFeature[i][h][w] = if (h < numRows) feature[i][h][w] else 0.0f
                }
            }
        }

        // Step 3: 转置 (HWC -> CHW)
        val chwFeature = Array(numCols) { Array(3) { FloatArray(padLen) } }
        for (c in 0 until numCols) {
            for (i in 0 until 3) {
                for (h in 0 until padLen) {
                    chwFeature[c][i][h] = paddedFeature[i][h][c]
                }
            }
        }

        println("Dimensions of chwFeature:")
        println("chwFeature First dimension : ${chwFeature.size}")
        println("chwFeature Second dimension : ${chwFeature[0].size}")
        println("chwFeature Third dimension : ${chwFeature[0][0].size}")
        return chwFeature
    }



    private fun convertToOnnxTensor(
        env: OrtEnvironment,
        feature: Array<Array<FloatArray>>
    ): OnnxTensor {
        val channels = 13
        val height = 3
        val width = 128

        val flattenedData = FloatArray(channels * 3 * width)
        var index = 0
        for (c in 0 until channels) {
            for (h in 0 until 3) {
                for (w in 0 until width) {
                    flattenedData[index++] = feature[c][h][w]
                }
            }
        }


        // 将数据转换为 FloatBuffer
        val buffer = FloatBuffer.wrap(flattenedData)

        // 定义 ONNX Tensor 的形状
        val shape = longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())

        // 创建 OnnxTensor
        return OnnxTensor.createTensor(env, buffer, shape)
    }


    private fun findMaxIndexAndScore(array: FloatArray): FloatArray {
        val resArray = FloatArray(2)
        var maxIndex = 0
        var maxValue = array[0]
        for ((index, value) in array.withIndex()) {
            if (value > maxValue) {
                maxValue = value
                maxIndex = index
            }
        }
        resArray[0] = maxIndex.toFloat()
        resArray[1] = maxValue

        return resArray
    }


    private fun softmax(input: FloatArray): FloatArray {
        // 1. 计算输入数组中的最大值
        val max = input.maxOrNull() ?: 0.0f

        // 2. 计算每个元素的指数
        val expArray = FloatArray(input.size) { i -> exp(input[i] - max) }

        // 3. 计算指数和
        val sumExp = expArray.sum()

        // 4. 计算softmax值
        return FloatArray(input.size) { i -> expArray[i] / sumExp }
    }



    // 执行 ONNX 推理
    private fun runOnnxInference(mfccFeatures: Array<Array<FloatArray>>): Int? {
        return try {
            // 将 MFCC 特征转换为 ONNX 输入张量
            val inputTensor = convertToOnnxTensor(ortEnv, mfccFeatures)

            val outputArray: FloatArray = inputTensor.getFloatBuffer().array()
            println(outputArray.toString())


            val inputName = ortSession.inputNames.iterator().next()
            // 准备 ONNX 模型的输入映射
            val inputMap = mapOf(inputName to inputTensor)

            // 运行 ONNX 推理
            val result = ortSession.run(inputMap)
            val scores = result[0].value as Array<FloatArray>

            val flattenedArray = scores.flatMap { it.asList() }.toFloatArray()
            val softmaxArray = softmax(flattenedArray)

            val resArray = findMaxIndexAndScore(softmaxArray)
            val maxIndex = resArray[0].toInt()
            val maxScore = resArray[1]
            // 获取推理结果
            Log.d("ONNX Inference", "HHH Scores: ${flattenedArray.contentToString()}")
            Log.d("ONNX Inference", "HHH MaxIndex: $maxIndex")
            Log.d("ONNX Inference", "HHH MaxScore: $maxScore")

            maxIndex
        } catch (e: OrtException) {
            Log.e("ONNX Inference", "Error during ONNX inference", e)
            null
        }
    }


    // 从 assets 文件夹读取 ONNX 模型文件
    private fun readAssetModel(filename: String): ByteArray {
        return assets.open(filename).use { it.readBytes() }
    }

}



