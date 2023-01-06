package com.example.testsegment2

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.ColorInt
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import com.example.testsegment2.databinding.ActivityMainBinding
import com.example.testsegment2.ml.Deeplabv3
import com.example.testsegment2.ml.Mobilenetv1
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.SegmentationMask
import com.google.mlkit.vision.segmentation.Segmenter
import org.tensorflow.lite.task.vision.segmenter.Segmentation
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter
import org.tensorflow.lite.task.vision.segmenter.OutputType
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val Gallerry_code_request=123
    private var imageSegmenter: ImageSegmenter? = null
    private val ALPHA_COLOR = 255
    val optionsBuilder = ImageSegmenter.ImageSegmenterOptions.builder()

    private var maskM:SegmentationMask?=null
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding=ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        imageSegmenter =
            ImageSegmenter.createFromFileAndOptions(
                this,
                "deeplabv3.tflite",
                optionsBuilder.build()
            )
        binding.apply {
            btnCamera.setOnClickListener{
                if(ContextCompat.checkSelfPermission(this@MainActivity,android.Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                    val intent= Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                    intent.type="image/*"
                    val mineType= arrayOf("image/jpg","image/jpeg")
                    intent.putExtra(Intent.EXTRA_MIME_TYPES,mineType)
                    intent.flags= Intent.FLAG_GRANT_READ_URI_PERMISSION
                    onResult.launch(intent)
                }else{
                    RequestPermission.launch(android.Manifest.permission.CAMERA)
                }
            }

            btnGradele.setOnClickListener{
                if(ContextCompat.checkSelfPermission(this@MainActivity,android.Manifest.permission.WRITE_EXTERNAL_STORAGE)== PackageManager.PERMISSION_GRANTED){

                }else{
                    RequestPermission.launch(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                }
            }
        }
    }




    @RequiresApi(Build.VERSION_CODES.O)
    private val RequestPermission=registerForActivityResult(ActivityResultContracts.RequestPermission()){
        if(it){
            TakePictureCamera.launch(null)
        }else{
            Toast.makeText(this,"Permission denine", Toast.LENGTH_LONG).show()
        }
    }
    // lauch camera and  take picture
    @RequiresApi(Build.VERSION_CODES.O)
    private val TakePictureCamera= registerForActivityResult(ActivityResultContracts.TakePicturePreview()){
            bitmap->
        if(bitmap!=null)
        {
            binding.imgRoot.setImageBitmap(bitmap)
            OutputMask(bitmap)
        }

    }

    @RequiresApi(Build.VERSION_CODES.O)
    private val onResult=registerForActivityResult(ActivityResultContracts.StartActivityForResult()){
        onResultReceiver(Gallerry_code_request,it)
    }
    @RequiresApi(Build.VERSION_CODES.O)
    private fun onResultReceiver(requestcode:Int, result: ActivityResult?){

        when(requestcode){
            Gallerry_code_request ->{
                if(result?.resultCode== RESULT_OK){
                    result.data?.data.let {
                            uri ->
                        val bitmap= BitmapFactory.decodeStream(contentResolver.openInputStream(uri!!))
                        binding.imgRoot.setImageBitmap(bitmap)
                        OutputMask(bitmap)
                    }
                }else{
                    Toast.makeText(this,"Error", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
    @RequiresApi(Build.VERSION_CODES.O)
    private fun OutputMask(image:Bitmap){   // main function
        optionsBuilder.setOutputType(OutputType.CATEGORY_MASK)
        imageSegmenter =
            ImageSegmenter.createFromFileAndOptions(
                this,
                "deeplabv3.tflite",
                optionsBuilder.build())
        val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-0 / 90))
                .build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val segmentResult = imageSegmenter?.segment(tensorImage)
        if (segmentResult != null) {
            setResult(segmentResult,image)
        }
    }
    private fun setResult(segmentResult:List<Segmentation>,image:Bitmap) {
        if(!segmentResult.isEmpty()){
            val colorLabels = segmentResult[0].coloredLabels.mapIndexed { index, coloredLabel ->
                ColorLabel(
                    index,
                    coloredLabel.getlabel(),
                    coloredLabel.argb
                )

            }
            val maskTensor = segmentResult[0].masks[0]
            val maskArray = maskTensor.buffer.array()
            val pixels = IntArray(maskArray.size)

            for (i in maskArray.indices) {
                val colorLabel = colorLabels[maskArray[i].toInt()].apply {
                    isExist = true
                }
                val color = colorLabel.getColor()
                pixels[i] = color
            }

            val newimage = Bitmap.createBitmap(
                pixels,
                maskTensor.width,
                maskTensor.height,
                Bitmap.Config.ARGB_8888
            )
            val bitmapResult=MergerBitmap(image,newimage)
            binding.imgDetected.setImageBitmap(bitmapResult)
            binding.imgDetected.isVisible=true
        }

    }
    inner class ColorLabel(
        val id: Int,
        val label: String,
        val rgbColor: Int,
        var isExist: Boolean = false
    ) {

        fun getColor(): Int {
            // Use completely transparent for the background color.
            return if (id == 0) Color.TRANSPARENT else Color.argb(
                ALPHA_COLOR,
                Color.red(rgbColor),
                Color.green(rgbColor),
                Color.blue(rgbColor)
            )
        }
    }
    private fun MergerBitmap(firstBM:Bitmap,sencondBM:Bitmap):Bitmap{
        val result = Bitmap.createBitmap(
            firstBM.getWidth(),
            firstBM.getHeight(),
            firstBM.getConfig()
        )
        val canvas = Canvas(result)
        canvas.drawBitmap(firstBM, 0f, 0f, null)
        canvas.drawBitmap(sencondBM, 0f, 0f, null)
        return result
    }
    @ColorInt
    private fun maskColorsFromByteBuffer(byteBuffer: ByteBuffer): IntArray {
        @ColorInt val colors =
            IntArray(maskM!!.width * maskM!!.height)
        for (i in 0 until maskM!!.width * maskM!!.height) {
            val backgroundLikelihood =  byteBuffer.float
            if (backgroundLikelihood > 0.9) {
                colors[i] =Color.argb(255, 0, 0, 0)
            } else if (backgroundLikelihood > 0.2) {
                // Linear interpolation to make sure when backgroundLikelihood is 0.2, the alpha is 0 and
                // when backgroundLikelihood is 0.9, the alpha is 128.
                // +0.5 to round the float value to the nearest int.
                val alpha = (182.9 * backgroundLikelihood - 36.6 + 0.5).toInt()
                colors[i] = Color.argb(alpha, 0, 0, 0)
//            }else {
//                    colors[i]=Color.argb(255,0,0,0)
//                }
        }}
        return colors
    }


    private fun performBW(originBitmap: Bitmap, maskBitmap: Bitmap): Bitmap? {
        val bmOut = Bitmap.createBitmap(
            originBitmap.width, originBitmap.height,
            originBitmap.config
        )
        val w = originBitmap.width
        val h = originBitmap.height
        val colors = IntArray(w * h)
        val colorsMask = IntArray(maskBitmap.width * maskBitmap.height)
        originBitmap.getPixels(colors, 0, w, 0, 0, w, h)
        maskBitmap.getPixels(colorsMask, 0, w, 0, 0, w, h)
        var pos: Int
        for (i in 0 until h) {
            for (j in 0 until w) {
                pos = i * w + j
                if (colorsMask[pos] == Color.argb(128, 255, 0, 255)) colors[pos] = Color.TRANSPARENT
            }
        }
        bmOut.setPixels(colors, 0, w, 0, 0, w, h)
        return bmOut
    }

}