package org.pytorch.demo.objectdetection;


import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;


import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;


import org.opencv.imgproc.Imgproc;
import org.pytorch.IValue;
//import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.opencv.android.OpenCVLoader;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import android.speech.tts.TextToSpeech;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity implements Runnable {
    private int mImageIndex = 0;
    private String[] mTestImages = {"3.png", "test2.jpg", "test3.png"};
    private String[] mTestImages1 = {"4.png", "3.png", "4.png"};
    private String[] Cls = {"person", "stairs", "box"};
    private String[] Position = {"left", "short left", "mid","short right","right"};

    private ImageView mImageView;
    private ResultView mResultView;
    private  TextToSpeech tts;
    private  TextToSpeech ttsResult;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Bitmap mBitmap1 = null;
    private Bitmap mBitmapir = null;
    private Bitmap mBitmapdepth = null;
    private boolean runflag = false;
    private Bitmap resultBitmap = null;
    private Bitmap resultcopyBitmap = null;
    private Mat finalZ_distance = null;
    private Matrix cropToFrameTransform;
    private Module mModule = null;
    private Bitmap inputx = null;
    private Bitmap inputdepth = null;
    private Bitmap catBitmap = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;
    private float rate = 1.0F;
    private boolean Wallflag = false;
    private long vioflag = 0;
    private YoloV5Classifier detector;
    private List<Classifier.Recognition> resultslite;
    private List<Classifier.Recognition> resultslite_Pre0;
    private ArrayList<Integer> stairscount = new ArrayList<>();
    private int framecount = 0;
    private int staircountint = 0;
    private boolean stairflag = false;
    private boolean closeResultview = false;
    private boolean stairflagplus = false;



    Handler handler = new Handler(Looper.getMainLooper());


    public static String assetFilePath(Context context, String assetName) throws IOException {

        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        OpenCVLoader.initDebug();
        super.onCreate(savedInstanceState);

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR) {
                    tts.setLanguage(Locale.ENGLISH);
                    tts.setSpeechRate(rate);
                }
            }
        });

        ttsResult = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR) {
                    ttsResult.setLanguage(Locale.CHINESE);
                    ttsResult.setSpeechRate(rate);
                }
            }
        });


        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
            mBitmap1 = BitmapFactory.decodeStream(getAssets().open(mTestImages1[mImageIndex]));
            mBitmapir = Bitmap.createBitmap(640, 480, mBitmap.getConfig());
            mBitmapdepth = Bitmap.createBitmap(640, 480, mBitmap.getConfig());
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
//        mImageView.setImageBitmap(catBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(("Test Image 1/3"));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.length));

                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages1[mImageIndex]));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });


        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                System.out.println("端口工作开始");
                //创建一个SocketTask对象
                SocketTask socketTask = new SocketTask();
                //执行这个任务
                socketTask.execute();
                if (runflag==true) {
                    Thread thread = new Thread(MainActivity.this);   //进入run（）
                    thread.start();
                }
            }
        });

        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
              final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
              startActivity(intent);
            }
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.mInputWidth;
                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.mInputHeight;

                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());

                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                int width = mBitmap.getWidth() + mBitmap1.getWidth();
                int height = Math.max(mBitmap.getHeight(), mBitmap1.getHeight());
                resultBitmap = Bitmap.createBitmap(width, height, mBitmap.getConfig());
                Canvas canvas = new Canvas(resultBitmap);

                canvas.drawBitmap(mBitmap, 0f, 0f, null);
                canvas.drawBitmap(mBitmap1, mBitmap.getWidth(), 0f, null);

//                mImageView.setImageBitmap(mBitmap1);

                mImageView.setImageBitmap(resultBitmap);

                Thread thread = new Thread(MainActivity.this);   //进入run（）
                thread.start();
            }
        });


        final String modelString = "best-fp16.tflite";
        stairscount.add(0);
        stairscount.add(0);
        stairscount.add(0);
        stairscount.add(0);
        stairscount.add(0);
        stairscount.add(0);
        stairscount.add(0);
        stairscount.add(0);




        try {
            detector = DetectorFactory.getDetector(getAssets(), modelString);    //得到权重并加载
            detector.useGpu();
//            detector.useNNAPI();
            detector.setNumThreads(5);
        } catch (final IOException e) {
            e.printStackTrace();
            Log.e("Object Detection", "Error reading assets", e);
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }


    private class SocketTask extends AsyncTask<Void, Void, Void> {
        //定义成员变量
        private Socket socket;
        private InputStream inputStream;
        private OutputStream outputStream;
        private int[][] cons;

        //重写doInBackground方法，在后台线程中调用
        @Override
        protected Void doInBackground(Void... voids) {
            try {
                ServerSocket ss = new ServerSocket(9000);
                System.out.println("启动服务器....");
                Socket s = ss.accept();
                System.out.println("客户端:" + s.getInetAddress().getCanonicalHostName() + " 已连接到服务器");
                byte[] tempData = new byte[0];
                int flag = 0;
                int bitscont = 0;
                int l1 = 0;
                int l2 = 0;
                int l3 = 0;
                int l4 = 0;
                int l6 = 0;
                int l7 = 0;
                int l8 = 0;
                int l9 = 0;
                int head = 130;   
                int MAX_BYTES = 2200000;
                int socketflag = 1;
                Mat imgs = null;
                Mat  rois = null;
                Mat roi;
                Mat z_distance = new Mat(480,640,CvType.CV_16U);
                Mat z_distance2 = null;
                int step = 255;
                int start_num = 128;
                int color_length = start_num + 3 * step + 128;
                Scalar color_scalar = new Scalar( color_length/255.0);
                Size imgsize =z_distance.size();

                Mat mul_color_scalar = new Mat(imgsize, CvType.CV_16U,color_scalar);
                Scalar color_scalar255 = new Scalar( 254);
                Mat mul_color_scalar255  = new Mat(imgsize, CvType.CV_16U,color_scalar255 );

                Scalar scalar_depth1 = new Scalar(1);
                Scalar scalar_depth2 = new Scalar(2);
                Scalar scalar_depth3 = new Scalar(3);


                Scalar scalar1 = new Scalar(0);
                Scalar scalar2 = new Scalar(start_num);
                Scalar scalar3 = new Scalar( start_num + step);
                Scalar scalar4 = new Scalar(start_num + 2 * step);
                Scalar scalar5 = new Scalar(start_num + 3 * step);

                Scalar scalar127 = new Scalar(127);
                Scalar scalar170 = new Scalar(170);
                Scalar scalar255 = new Scalar(255);

                Scalar scalar_2step_start_num = new Scalar(2 * step + start_num);

                Scalar scalar_start_num = new Scalar(start_num);
                Scalar scalar_3step_start_num = new Scalar(3 * step + start_num);

                Scalar scalar_start_num_step = new Scalar(start_num + step);
                Scalar scalar128 = new Scalar(128);
                Mat add170 = new Mat(imgsize, CvType.CV_16U,scalar170);

                Mat add127 = new Mat(imgsize, CvType.CV_16U,scalar127);
                Mat add127temp = new Mat(imgsize, CvType.CV_16U);
                Mat add255= new Mat(imgsize, CvType.CV_16U,scalar255);

                Mat add_2step_start_num = new Mat(imgsize, CvType.CV_16U,scalar_2step_start_num);
                Mat add_2step_start_num_temp = new Mat(imgsize, CvType.CV_16U);

                Mat add_start_num = new Mat(imgsize, CvType.CV_16U,scalar_start_num);
                Mat add_start_num_temp = new Mat(imgsize, CvType.CV_16U);

                Mat add_3step_start_num = new Mat(imgsize, CvType.CV_16U,scalar_3step_start_num);
                Mat add_3step_start_num_temp = new Mat(imgsize, CvType.CV_16U);

                Mat add_start_num_step = new Mat(imgsize, CvType.CV_16U,scalar_start_num_step);
                Mat add_start_num__step_temp = new Mat(imgsize, CvType.CV_16U);
                Mat add128= new Mat(imgsize, CvType.CV_16U,scalar128);

                Mat img1 = new Mat(imgsize, CvType.CV_16U,scalar1);
                Mat img2 = new Mat(imgsize, CvType.CV_16U,scalar2);
                Mat img3 = new Mat(imgsize, CvType.CV_16U,scalar3);
                Mat img4 = new Mat(imgsize, CvType.CV_16U,scalar4);
                Mat img5 = new Mat(imgsize, CvType.CV_16U,scalar5);

                Mat output_LE1 = new Mat(imgsize, CvType.CV_16U);
                Mat output_GT1 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE2 = new Mat(imgsize, CvType.CV_16U);
                Mat output_GT2 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE3 = new Mat(imgsize, CvType.CV_16U);
                Mat output_GT3 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE4 = new Mat(imgsize, CvType.CV_16U);
                Mat output_GT4 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE5 = new Mat(imgsize, CvType.CV_16U);
                Mat output5 = new Mat(imgsize, CvType.CV_16U);

                Mat depth1 = new Mat(imgsize, CvType.CV_8U,scalar_depth1);
                Mat depth2 = new Mat(imgsize, CvType.CV_8U,scalar_depth2);
                Mat depth3 = new Mat(imgsize, CvType.CV_8U,scalar_depth3);
                Mat depth4 = new Mat(imgsize, CvType.CV_16U);
                Mat output_de1 = new Mat(imgsize, CvType.CV_8U);
                Mat output_de2 = new Mat(imgsize, CvType.CV_8U);
                Mat output_de3 = new Mat(imgsize, CvType.CV_8U);
                Mat output_de12 = new Mat(imgsize, CvType.CV_8U);
                Mat output_de23 = new Mat(imgsize, CvType.CV_8U);
                Mat output_de = new Mat(imgsize, CvType.CV_16U);
                Mat output_de4 = new Mat(imgsize, CvType.CV_16U);

                Mat output1,output10 = new Mat(imgsize, CvType.CV_16U);
                Mat output2,output20 = new Mat(imgsize, CvType.CV_16U);
                Mat output3,output30 = new Mat(imgsize, CvType.CV_16U);
                Mat output4,output40 = new Mat(imgsize, CvType.CV_16U);
                Mat output50 = new Mat(imgsize, CvType.CV_16U);

                Mat color_data1_add12 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data2_add12 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data1_add23 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data2_add23 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data3_add12 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data3_add23 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data1_add34 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data2_add34 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data3_add34 = new Mat(imgsize, CvType.CV_16U);

                Mat color_data1 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data2 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data3 = new Mat(imgsize, CvType.CV_16U);

                Mat output_de10 = new Mat(imgsize, CvType.CV_16U);
                Mat output_de40 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE10=new Mat();
                Mat output100 = new Mat(imgsize, CvType.CV_16U);
                Mat output200 = new Mat(imgsize, CvType.CV_16U);
                Mat output300 = new Mat(imgsize, CvType.CV_16U);
                Mat output400 = new Mat(imgsize, CvType.CV_16U);
                Mat output500 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE100 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE_10 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE_100 = new Mat(imgsize, CvType.CV_16U);
                Mat output_LE_1000 = new Mat(imgsize, CvType.CV_16U);
                Mat color_data = new Mat(z_distance.size(), CvType.CV_16UC3);
                Mat color_data22 = new Mat(z_distance.size(), CvType.CV_8UC3);

                Thread thread = new Thread(MainActivity.this);   //进入run（）
                thread.start();

                long Loopstart,recivestart=0,reciveend=0,decoderstart=0;
                while (true) {
                    flag = 0;
                    Loopstart = System.currentTimeMillis();
                    while (true) {
                        byte[] data = new byte[MAX_BYTES];
                        int len = s.getInputStream().read(data);

                        if (len == -1) {
                            socket.close();
                            break;
                        }
                        tempData = Arrays.copyOf(tempData, tempData.length + len);//扩展tempData数组
                        System.arraycopy(data, 0, tempData, tempData.length - len, len);//

                        if (flag == 0) {
                            decoderstart= System.currentTimeMillis();
                            l1 = Integer.parseInt(new String(Arrays.copyOfRange(tempData, 0, 15)).trim());
                            bitscont = bitscont + l1;
                            l2 = Integer.parseInt(new String(Arrays.copyOfRange(tempData, 15, 30)).trim());
                            l3 = Integer.parseInt(new String(Arrays.copyOfRange(tempData, 30, 45)).trim());
                            l4 = Integer.parseInt(new String(Arrays.copyOfRange(tempData, 45, 60)).trim());
                            l6 = Integer.parseInt(new String(Arrays.copyOfRange(tempData, 60, 75)).trim());
                            l7 = Integer.parseInt(new String(Arrays.copyOfRange(tempData, 75, 95)).trim());
                            l8 = Integer.parseInt(new String(Arrays.copyOfRange(tempData, 95, 115)).trim());
                            l9 = Integer.parseInt(new String(Arrays.copyOfRange(tempData, 115, 130)).trim());
                            flag = 1;
                        }

                        while (tempData.length >= 2.5 * l1) {
                            byte[] temp_datasource = new byte[0];
                            temp_datasource = Arrays.copyOf(temp_datasource, temp_datasource.length + l1);
                            System.arraycopy(tempData, 0, temp_datasource, 0, l1);
                            int l1source = l1;
                            int l2source = l2;
                            int l3source = l3;
                            int l4source = l4;
                            int l6source = l6;
                            int l7source = l7;
                            int l8source = l8;
                            int l9source = l9;
                            byte[] tempDatacache = new byte[tempData.length - l1];
                            System.arraycopy(tempData, l1, tempDatacache, 0, tempData.length - l1);
                            tempData = tempDatacache;//temp_data=temp_datasource+temp_data
                            //System.arraycopy(tempData, l1, tempData, 0, tempData.length - l1);
                            l1 = Integer.parseInt(new String(tempData, 0, 15, "ascii").trim());
                            bitscont = bitscont + l1;
                            l2 = Integer.parseInt(new String(tempData, 15, 15, "ascii").trim());
                            l3 = Integer.parseInt(new String(tempData, 30, 15, "ascii").trim());
                            l4 = Integer.parseInt(new String(tempData, 45, 15, "ascii").trim());
                            l6 = Integer.parseInt(new String(tempData, 60, 15, "ascii").trim());
                            l7 = Integer.parseInt(new String(tempData, 75, 20).trim());
                            l8 = Integer.parseInt(new String(tempData, 95, 20).trim());
                            l9 = Integer.parseInt(new String(tempData, 115, 15).trim());
                            long datalonger25 = System.currentTimeMillis();
//                            System.out.println("数据长度大于2.5l时截取后的时间：" + (datalonger25));
                            if (tempData.length < l1) {
//                                decoderstart = System.currentTimeMillis();
                                byte[] tempDatanew = new byte[temp_datasource.length + tempData.length];
                                System.arraycopy(temp_datasource, 0, tempDatanew, 0, temp_datasource.length);
                                System.arraycopy(tempData, 0, tempDatanew, temp_datasource.length, tempData.length);
                                tempData = tempDatanew;//temp_data=temp_datasource+temp_data

                                bitscont = bitscont - l1;
                                l1 = l1source;
                                l2 = l2source;
                                l3 = l3source;
                                l4 = l4source;
                                l6 = l6source;
                                l7 = l7source;
                                l8 = l8source;
                                l9 = l9source;
//                                long datashorter1 = System.currentTimeMillis();
//                                System.out.println("数据长度小于l时保存后的时间：" + (datashorter1));
                                break;
                            }
                        }

                        if (tempData.length >= l1) {
                            recivestart=System.currentTimeMillis();
                            String con = new String(tempData, head, l2, "ascii");
                            byte[] image = Arrays.copyOfRange(tempData, head + l2, head + l2 + l3);
                            rois = Imgcodecs.imdecode(new MatOfByte(image), Imgcodecs.IMREAD_COLOR);
                            image = Arrays.copyOfRange(tempData, head + l2 + l3, head + l2 + l3 + l4);
                            imgs = Imgcodecs.imdecode(new MatOfByte(image), Imgcodecs.IMREAD_COLOR);
                            image = Arrays.copyOfRange(tempData, l3 + head + l2 + l4, l3 + head + l2 + l4 + l6);
                            z_distance2 = Imgcodecs.imdecode(new MatOfByte(image), Imgcodecs.IMREAD_GRAYSCALE);

                            String[] conList = con.split(" ");
                            int[] cons = new int[conList.length];
                            for (int i = 0; i < conList.length; i++) {
                                cons[i] = Integer.parseInt(conList[i]);
//                                System.out.println("cons：" + cons[i]);
                            }

                            roi = imgs.submat(new Rect(cons[1], cons[0], cons[3], cons[2]));
                            rois.copyTo(roi);

                            Imgproc.cvtColor(imgs, imgs, Imgproc.COLOR_BGR2GRAY);
                            Imgproc.equalizeHist(imgs, imgs);
                            Imgproc.cvtColor(imgs, imgs, Imgproc.COLOR_GRAY2BGR);
                            tempData = Arrays.copyOfRange(tempData, l1, tempData.length);
                            reciveend=System.currentTimeMillis();

                            break;
                        }
                    }
                    long start2,end2;

                    // z_distance=Imgcodecs.imread("0008.jpg");
                    {
                        start2 = System.currentTimeMillis();
                        z_distance2.assignTo(z_distance, 2);
                        Mat img = z_distance.mul(mul_color_scalar);
//1通道
                        Core.add(add127, img, add127temp);
                        Core.subtract(add_2step_start_num, img, add_2step_start_num_temp);
//2通道
                        Core.subtract(img, add_start_num, add_start_num_temp);
                        Core.subtract(add_3step_start_num, img, add_3step_start_num_temp);
//3通道
                        Core.subtract(img, add_start_num_step, add_start_num__step_temp);

                        Core.compare(img, img1, output_LE1, Core.CMP_LE);//小于等于
                        Core.compare(img, img1, output_GT1, Core.CMP_GT);//大于
                        Core.compare(img, img2, output_LE2, Core.CMP_LE);//小于等于
                        Core.compare(img, img2, output_GT2, Core.CMP_GT);
                        Core.compare(img, img3, output_LE3, Core.CMP_LE);//小于等于
                        Core.compare(img, img3, output_GT3, Core.CMP_GT);
                        Core.compare(img, img4, output_LE4, Core.CMP_LE);//小于等于
                        Core.compare(img, img4, output_GT4, Core.CMP_GT);
                        Core.compare(img, img5, output_LE5, Core.CMP_LE);//小于等于
                        Core.compare(img, img5, output5, Core.CMP_GT);//大于
                        output1 = output_GT1.mul(output_LE2);
                        output2 = output_GT2.mul(output_LE3);
                        output3 = output_GT3.mul(output_LE4);
                        output4 = output_GT4.mul(output_LE5);

                        Core.compare(z_distance2, depth1, output_de1, Core.CMP_EQ);//等于
                        Core.compare(z_distance2, depth2, output_de2, Core.CMP_EQ);//等于
                        Core.compare(z_distance2, depth3, output_de3, Core.CMP_EQ);//等于
                        Core.add(output_de1, output_de2, output_de12);
                        Core.add(output_de12, output_de3, output_de23);
                        output_de23.assignTo(output_de, 2);

                        Core.subtract(output_de, mul_color_scalar255, output_de10);

                        Core.compare(output_de, img1, depth4, Core.CMP_EQ);//等于
                        depth4.assignTo(output_de4, 2);


                        Core.subtract(output_de4, mul_color_scalar255, output_de40);


                        output1.assignTo(output10, 2);
                        output2.assignTo(output20, 2);
                        output3.assignTo(output30, 2);
                        output4.assignTo(output40, 2);
                        output5.assignTo(output50, 2);

                        output_LE1.assignTo(output_LE10, 2);

                        Core.subtract(output10, mul_color_scalar255, output100);
                        Core.subtract(output20, mul_color_scalar255, output200);
                        Core.subtract(output30, mul_color_scalar255, output300);
                        Core.subtract(output40, mul_color_scalar255, output400);
                        Core.subtract(output50, mul_color_scalar255, output500);
                        Core.subtract(output_LE10, mul_color_scalar255, output_LE100);

                        Core.compare(output_LE10, img1, output_LE_10, Core.CMP_EQ);//等于
                        output_LE_10.assignTo(output_LE_100, 2);
                        Core.subtract(output_LE_100, mul_color_scalar255, output_LE_1000);

                        Core.add(output100.mul(add127temp), output200.mul(add255), color_data1_add12);
                        Core.add(color_data1_add12, output300.mul(add_2step_start_num_temp), color_data1_add23);
                        Core.add(color_data1_add23.mul(output_LE_1000), output_LE100.mul(img1), color_data1_add34);
                        Core.add(color_data1_add34.mul(output_de40), output_de10.mul(add127), color_data1);

                        Core.add(output200.mul(add_start_num_temp), output300.mul(add255), color_data2_add12);
                        Core.add(color_data2_add12, output400.mul(add_3step_start_num_temp), color_data2_add23);
                        Core.add(color_data2_add23.mul(output_LE_1000), output_LE100.mul(img1), color_data2_add34);
                        Core.add(color_data2_add34.mul(output_de40), output_de10.mul(img1), color_data2);

                        Core.add(output300.mul(add_start_num__step_temp), output400.mul(add255), color_data3_add12);
                        Core.add(color_data3_add12, output500.mul(add128), color_data3_add23);
                        Core.add(color_data3_add23.mul(output_LE_1000), output_LE100.mul(img1), color_data3_add34);
                        Core.add(color_data3_add34.mul(output_de40), output_de10.mul(add170), color_data3);

                        List<Mat> color_data0 = new ArrayList<>();
                        color_data0.add(color_data3);
                        color_data0.add(color_data2);
                        color_data0.add(color_data1);

                        Core.merge(color_data0, color_data);
                        color_data.assignTo(color_data22, CvType.CV_8UC3);


                        end2 = System.currentTimeMillis();
//                        System.out.println("depth ray to rgb start time:" + start2 + "; end time:" + end2 + "; Run Time:" + (end2 - start2) + "(ms)");
                    }

                    long decodend = System.currentTimeMillis();
                    System.out.println("接收的总字节数  " + bitscont / 1024 + " KB");
                    System.out.println("解码时间：" + (reciveend-recivestart));
                    System.out.println("开始时间戳：" + (decoderstart));
                    System.out.println("彩色转换结束时间戳：" + (decodend));
                    System.out.println("接收一个包开始 加解码 加彩色转换 用时：" + (decodend - decoderstart));
                    System.out.println("所有接收加解码加彩色转换用时：" + (decodend - Loopstart));
                    System.out.println("**************************************************************************************");

                    Utils.matToBitmap(imgs, mBitmapir);
                    Utils.matToBitmap(color_data22, mBitmapdepth);

                    mImageView.setImageBitmap(mBitmapdepth);

                    //拼接图象
                    int width = mBitmapir.getWidth() + mBitmapdepth.getWidth();
                    int height = Math.max(mBitmapir.getHeight(), mBitmapdepth.getHeight());
                    catBitmap = Bitmap.createBitmap(width, height, mBitmapir.getConfig());
                    Canvas canvas = new Canvas(catBitmap);

                    canvas.drawBitmap(mBitmapir, 0f, 0f, null);
                    canvas.drawBitmap(mBitmapdepth, mBitmapir.getWidth(), 0f, null);
//                    mImageView.setImageBitmap(catBitmap);
//
                    finalZ_distance = z_distance;
//                    Imgproc.cvtColor(z_distance, z_distance, Imgproc.COLOR_BGR2GRAY);
                    Scalar mean = Core.mean(z_distance);
                    int avgDistance = (int)(mean.val[0]);
                    System.out.println("aveDistance:"+avgDistance);
                    runflag=true;
                }


            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }


    }




    @Override
    public void run() {
        long start, end;
//        streaming线程
        while (true) {
            if (runflag) {
                start = System.currentTimeMillis();
                resultslite = detector.recognizeImage(catBitmap);
                int frame_threshold = 6;
                if (framecount > 0 && framecount < frame_threshold) {
                    framecount++;
                }
                stairflagplus = false;
//                start = System.currentTimeMillis();
                for (final Classifier.Recognition result : resultslite) {
                    int cls = result.getDetectedClass();
                    if (cls == 4) {
                        if (framecount == 0) {   //启动窗口
                            framecount = 1;
                        }
                        staircountint = 0;
                        for (int i : stairscount) {
                            staircountint += i;
                        }
                        if ( staircountint < frame_threshold) {
                            stairscount.remove(0);
                            stairscount.add(1);
                        }
                        stairflagplus = true;
                        staircountint = 0;
                        for (int i : stairscount) {
                            staircountint += i;
                        }
                        if (staircountint / framecount > 0.2 && framecount > 2 && stairflag == false) {
//                            System.out.println(staircount/framecount);
                            stairflag = true;
                            String vio = "Attention, staris start";
//                            closeResultview = true;
//                            tts.speak(vio, TextToSpeech.QUEUE_FLUSH, null);
                        }
                        break;
                    }
                    if (cls == 0 && stairflag == true) {
                        staircountint = 0;
                        for (int i : stairscount) {
                            staircountint += i;
                        }
                        if ( staircountint < frame_threshold) {
                            stairscount.remove(0);
                            stairscount.add(1);
                        }
                        stairflagplus = true;
                        break;
                    }
                }
                for (final Classifier.Recognition result : resultslite) {
                    int cls = result.getDetectedClass();
                    if (cls == 0 && stairflag == true && System.currentTimeMillis() - vioflag >= 1300){
                        String vio = "Careful, Person ahead";
//                        tts.speak(vio, TextToSpeech.QUEUE_FLUSH, null);
                        vioflag = System.currentTimeMillis();
                    }
                }
                staircountint = 0;
                for (int i : stairscount) {
                    staircountint += i;
                }
                if (stairflagplus == false && staircountint > 0) {
                    stairscount.remove(0);
                    stairscount.add(0);
                }
                if (framecount > 0) {
                    staircountint = 0;
                    for (int i : stairscount) {
                        staircountint += i;
                    }
                    System.out.println(stairscount);
                    if (staircountint / framecount < 0.2 && framecount == frame_threshold && stairflag == true) {
                        //结束
                        framecount = 0;
                        for (int i = 0; i < stairscount.size(); i++) {
                            stairscount.set(i, 0);
                        }
                        stairflag = false;
                        String vio = "Attention, staris end";
//                        closeResultview = false;
//                        tts.speak(vio, TextToSpeech.QUEUE_FLUSH, null);
                    }
                }

                end = System.currentTimeMillis();

                System.out.println("start time:" + start + "; end time:" + end + "; Run Time:" + (end - start) + "(ms)");
                runflag = false;
                runOnUiThread(() -> {
                    mResultView.setResults(resultslite, resultslite_Pre0, finalZ_distance, ttsResult);
                    mResultView.invalidate();
                    mResultView.setVisibility(View.VISIBLE);
                    resultslite_Pre0 = resultslite;
                });
            }

            Wallflag=false;
        }

    }

}

