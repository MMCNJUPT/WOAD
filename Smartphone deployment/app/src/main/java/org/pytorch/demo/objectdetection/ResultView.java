// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import static java.lang.Math.abs;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.CornerPathEffect;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PathEffect;
import android.graphics.RectF;
import android.speech.tts.TextToSpeech;
import android.util.AttributeSet;
import android.view.View;
import android.widget.Toast;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.tensorflow.lite.examples.detection.tflite.Classifier;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.net.DatagramSocket;
import java.net.DatagramPacket;
import java.net.InetAddress;
import java.util.Locale;


public class ResultView<SocketTask, SocketFeedback> extends View {

    private final static int TEXT_X = 40;
    private final static int TEXT_Y = 35;
    private final static int TEXT_WIDTH = 260;
    private final static int TEXT_HEIGHT = 50;
    private Paint mPaintRectangle;
    private Paint tagsquarL_Green;
    private Paint tagsquarM_Green;
    private Paint tagsquarR_Green;
    private Paint tagsquarL_Yellow;
    private Paint tagsquarM_Yellow;
    private Paint tagsquarR_Yellow;
    private Paint tagsquarL_Red;
    private Paint tagsquarM_Red;
    private Paint tagsquarR_Red;
    private Paint O_tagsquarL;
    private Paint O_tagsquarM;
    private Paint O_tagsquarR;
    private TextToSpeech tts;
    private Paint mPaintText;
//    private ArrayList<Result> mResults;
    private List<Classifier.Recognition> mResults;

    private List<Classifier.Recognition> mResults_Pre;
    private Mat Z_distance;
    private Socket feedback;
    private String Positionvio[] = {"左边   ","左边   ","中间   ","中间   ","右边   ","右边   "};
    private String Cls[] = {"person","stairs","box"};
    private String DistanceStr[] = { " ","Danger  ","Attention  ","Alert  "};
    private long vioflag = 0;

    DatagramSocket ds;
    InetAddress loc;
    String str_send;


    public ResultView(Context context) {
        super(context);
    }

    public ResultView(Context context, AttributeSet attrs){
        super(context, attrs);
        mPaintRectangle = new Paint();
        mPaintRectangle.setColor(Color.YELLOW);
        mPaintText = new Paint();
        tagsquarL_Green = new Paint();
        tagsquarL_Green.setColor(Color.GREEN);
        tagsquarM_Green = new Paint();
        tagsquarM_Green.setColor(Color.GREEN);
        tagsquarR_Green = new Paint();
        tagsquarR_Green.setColor(Color.GREEN);
        tagsquarL_Yellow = new Paint();
        tagsquarL_Yellow.setColor(Color.YELLOW);
        tagsquarM_Yellow = new Paint();
        tagsquarM_Yellow.setColor(Color.YELLOW);
        tagsquarR_Yellow = new Paint();
        tagsquarR_Yellow.setColor(Color.YELLOW);
        tagsquarL_Red = new Paint();
        tagsquarL_Red.setColor(Color.RED);
        tagsquarM_Red = new Paint();
        tagsquarM_Red.setColor(Color.RED);
        tagsquarR_Red = new Paint();
        tagsquarR_Red.setColor(Color.RED);
        O_tagsquarL = new Paint();
        O_tagsquarL.setColor(Color.GRAY);
        O_tagsquarM = new Paint();
        O_tagsquarM.setColor(Color.GRAY);
        O_tagsquarR = new Paint();
        O_tagsquarR.setColor(Color.GRAY);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        O_tagsquarL.setStrokeWidth(8);   //设置画笔宽度
        O_tagsquarL.setStyle(Paint.Style.FILL);    //描边模式
        canvas.drawCircle(375, 1350, 50, O_tagsquarL);
        O_tagsquarM.setStrokeWidth(8);   //设置画笔宽度
        O_tagsquarM.setStyle(Paint.Style.FILL);    //描边模式
        canvas.drawCircle(750, 1350, 50, O_tagsquarM);
        O_tagsquarR.setStrokeWidth(8);   //设置画笔宽度
        O_tagsquarR.setStyle(Paint.Style.FILL);    //描边模式
        canvas.drawCircle(1125, 1350, 50, O_tagsquarR);
        if (mResults == null) return;
        int det = mResults.size();
        int detx = 1;
        if (det == 0) {
            detx = 1;   //三维数组，第一个值为distance，第二个值为position，第三个值为flag；当前数组为全零
        }else{
            detx = det;
        }
        int[][] Result_Array = new int[detx][4];
        int i = 0;
        for (final Classifier.Recognition result : mResults) {
//                //////计算距离模块

            final RectF location = result.getLocation();
            int cls = result.getDetectedClass();   //0是人，1是楼梯，2是箱子
            Result_Array[0][3] = cls;
            Float con = result.getConfidence();
            int centerx = (int) ((int) ((location.left + location.right) / 2));
            int centery = (int) ((int) ((location.top + location.bottom) / 2));
            int position = (int) centerx / 213;    //三个位置
            int positionvio = (int) centerx / 106;
            Result_Array[i][1] = positionvio;
            int distance = 0;

            if (centerx < 10 || centery < 10) {
                distance = (int) Z_distance.get(centerx, centery)[0];   //输入当前指针值。
            } else {
                Mat centerdistance = Z_distance.submat(centery - 10, centery + 10, centerx - 10, centerx + 10);
                Scalar average = Core.mean(centerdistance);
                double v = 7500 / 255.0;
                distance = (int) ((int) v * average.val[0] / 10);
//                System.out.println("本次计算得到的diatace数值为："+distance);
            }
            Result_Array[i][0] = distance;
            if (Result_Array[0][0]==0){
                Result_Array[0][0] = 0;
            }else if(Result_Array[0][0]<=50) {
                Result_Array[0][0] = 1;
            }else if(50<Result_Array[0][0]&&Result_Array[0][0]<=100){
                Result_Array[0][0] = 1;
            }else if(100<Result_Array[0][0]&&Result_Array[0][0]<=150){
                Result_Array[0][0] = 2;
            }else if(150<Result_Array[0][0]&&Result_Array[0][0]<=200){
                Result_Array[0][0] = 2;
            }else if(200<Result_Array[0][0]&&Result_Array[0][0]<=250){
                Result_Array[0][0] = 2;
            }else if(250<Result_Array[0][0]&&Result_Array[0][0]<=300){
                Result_Array[0][0] = 3;
            }else if(300<Result_Array[0][0]&&Result_Array[0][0]<=350){
                Result_Array[0][0] = 3;
            }else if(350<Result_Array[0][0]&&Result_Array[0][0]<=400){
                Result_Array[0][0] = 3;
            }else if(400<Result_Array[0][0]&&Result_Array[0][0]<=450){
                Result_Array[0][0] = 3;
            }else if(450<Result_Array[0][0]&&Result_Array[0][0]<=500){
                Result_Array[0][0] = 3;
            }else if(500<Result_Array[0][0]&&Result_Array[0][0]<=550){
                Result_Array[0][0] = 3;
            }
            i++; //此时当前目标的所有信息已经读取，后续无二维数组操作。进行增值
            if (cls == 4) {
                //三个一起亮
                tagsquarL_Red.setStrokeWidth(8);   //设置画笔宽度
                tagsquarL_Red.setStyle(Paint.Style.FILL);    //描边模式
                canvas.drawCircle(375, 1350, 50, tagsquarL_Red);
                tagsquarM_Red.setStrokeWidth(8);   //设置画笔宽度
                tagsquarM_Red.setStyle(Paint.Style.FILL);    //描边模式
                canvas.drawCircle(750, 1350, 50, tagsquarM_Red);
                tagsquarR_Red.setStrokeWidth(8);   //设置画笔宽度
                tagsquarR_Red.setStyle(Paint.Style.FILL);    //描边模式
                canvas.drawCircle(1125, 1350, 50, tagsquarR_Red);
            } else {
                switch (position) {
                    case 0:
                        switch (Result_Array[0][0]){
                            case 1:
                                tagsquarL_Red.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarL_Red.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(375, 1350, 50, tagsquarL_Red);
                                break;
                            case 2:
                                tagsquarL_Yellow.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarL_Yellow.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(375, 1350, 50, tagsquarL_Yellow);
                                break;
                            case 3:
                                tagsquarL_Green.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarL_Green.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(375, 1350, 50, tagsquarL_Green);
                                break;
                        }
                        break;
                    case 1:
                        switch (Result_Array[0][0]){
                            case 1:
                                tagsquarM_Red.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarM_Red.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(750, 1350, 50, tagsquarM_Red);
                                break;
                            case 2:
                                tagsquarM_Yellow.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarM_Yellow.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(750, 1350, 50, tagsquarM_Yellow);
                                break;
                            case 3:
                                tagsquarM_Green.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarM_Green.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(750, 1350, 50, tagsquarM_Green);
                                break;
                        }
                        break;
                    case 2:
                        switch (Result_Array[0][0]){
                            case 1:
                                tagsquarR_Red.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarR_Red.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(1125, 1350, 50, tagsquarR_Red);
                                break;
                            case 2:
                                tagsquarR_Yellow.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarR_Yellow.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(1125, 1350, 50, tagsquarR_Yellow);
                                break;
                            case 3:
                                tagsquarR_Green.setStrokeWidth(8);   //设置画笔宽度
                                tagsquarR_Green.setStyle(Paint.Style.FILL);    //描边模式
                                canvas.drawCircle(1125, 1350, 50, tagsquarR_Green);
                                break;
                        }
                        break;
                }
            }

            mPaintRectangle.setStrokeWidth(8);   //设置画笔宽度
            mPaintRectangle.setStyle(Paint.Style.STROKE);    //描边模式
            mPaintRectangle.setStrokeJoin(Paint.Join.BEVEL);
//            mPaintRectangle.setAlpha(99);
//            mPaintRectangle.setStrokeAlias(true);
            PathEffect pathEffect = new CornerPathEffect(20);
            mPaintRectangle.setPathEffect(pathEffect);
            location.top = (float) ((location.top * 1.33 * 1.73 + 160));
            location.bottom = (float) ((location.bottom * 1.33 * 1.73 + 160));
            location.left = (float) (location.left * 1.77 * 1.3);
            location.right = (float) (location.right * 1.77 * 1.3);
            canvas.drawRect(location, mPaintRectangle);
            ////打标签
            Path mPath = new Path();
//            RectF mRectF = new RectF(location.left, location.top, location.left + TEXT_WIDTH,  location.top + TEXT_HEIGHT);
//            mPath.addRect(mRectF, Path.Direction.CW);
//            mPaintText.setColor(Color.MAGENTA);
//            canvas.drawPath(mPath, mPaintText);

            mPaintText.setColor(Color.WHITE);
            mPaintText.setStrokeWidth(0);
            mPaintText.setStyle(Paint.Style.FILL);
            mPaintText.setTextSize(50);
            canvas.drawText(String.format("%s %d", result.getDetectedClass(), distance), location.left + TEXT_X, location.top + TEXT_Y, mPaintText);
//            canvas.drawText(String.format("%s %s", result.getDetectedClass(), distance), location.left + TEXT_X, location.top + TEXT_Y, mPaintText);


            String Distance = "" + distance;
            ////准备数据以及发送
//            System.out.println("把distace转为字符串后："+Distance+"   长度为："+Distance.length()+"   Byte值为："+Distance.getBytes().length);
            str_send = null;
            if (Distance.length() >= 4) {
                // 字符串太长了
                System.out.println("当前Distance长度过长，进行修剪");
                str_send = Distance.substring(0, 4);  //当长度大于四时取前四个值
            }
            if (Distance.length() < 4) {
                // str_send
                System.out.println("当前Distance长度不够，进行补充");
                str_send = String.format("%04d", Integer.parseInt(Distance));  //当长度小于四时在左侧填充0直到长度为4
            }
            System.out.println("补充修改后的str_send为：" + str_send + "   长度为：" + str_send.length() + "   Byte长度为：" + str_send.getBytes().length);
            str_send = cls + str_send;
            System.out.println("增加类别标志位后的str_send为：" + str_send + "   长度为：" + str_send.length() + "   Byte长度为：" + str_send.getBytes().length);
            switch (position) {
                case 0:
                    str_send = "0" + str_send;
                    break;
                case 1:
                    str_send = "1" + str_send;
                    break;
                case 2:
                    str_send = "2" + str_send;
                    break;
            }
            //str_send 为位置+种类+距离1+1+4
            //震动反馈代码Socket
            String Ssend = str_send;
            System.out.println("最终准备发送的str_send为：" + str_send + "   长度为：" + str_send.length() + "   Byte长度为：" + str_send.getBytes().length);
            Thread threadsend = new Thread(() -> {
                try {
                    ds = new DatagramSocket();
                    loc = InetAddress.getByName("192.168.84.175");
                    if (str_send != null) {
                        DatagramPacket dp_send = new DatagramPacket(Ssend.getBytes(), 6, loc, 9001);
                        ds.send(dp_send);
                        System.out.println("最终准备发送的Ssend为：" + Ssend + "   长度为：" + Ssend.length());
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
            threadsend.start();
        }

        System.out.println("排序前的结果："+Arrays.deepToString(Result_Array));
        Arrays.sort(Result_Array,new Comparator<int[]>(){
            public int compare(int[] a,int[] b){
                return a[0]-b[0];
            }
        });
//        //语音播报模块(英文)
//        System.out.println("排序后的结果："+ Arrays.deepToString(Result_Array));
//
//        String outvio = "" + DistanceStr[Result_Array[0][0]] + Positionvio[Result_Array[0][1]];
//        if (vioflag == 0 || System.currentTimeMillis() - vioflag >= 600 && Result_Array[0][0] !=0 && Result_Array[0][3]!=4) {
//            System.out.println("播报语音值："+outvio);
//            tts.speak(outvio, TextToSpeech.QUEUE_FLUSH, null);
//            vioflag = System.currentTimeMillis();
//        }


        //语音播报模块(简化中文)
        System.out.println("排序后的结果："+ Arrays.deepToString(Result_Array));
//        if (Result_Array[0][0] == 0) {
//            String outvio = "" + DistanceStr[Result_Array[0][0]] + Positionvio[Result_Array[0][1]];
        String outvio = "危险";
        outvio =  Positionvio[Result_Array[0][1]] + outvio;
        if (vioflag == 0 || System.currentTimeMillis() - vioflag >= 1000 && Result_Array[0][0] != 0 && Result_Array[0][3] != 4) {
            System.out.println("播报语音值：" + outvio);
            tts.speak(outvio, TextToSpeech.QUEUE_FLUSH, null);
            vioflag = System.currentTimeMillis();
        }
//        }
    }


    public void setResults(List<Classifier.Recognition> results, List<Classifier.Recognition> results_Pre, Mat finalZ_distance,TextToSpeech ttResult) {
        tts = ttResult;
        mResults = results;
        mResults_Pre = results_Pre;
        Z_distance = finalZ_distance;
    }
}
