You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1Kava0aKGvZWK7KlZPpPcejlNSpcZgbpT/view?usp=sharing).

### Environment
* Python 3.8 
* OpenCV-java 4.5.4
* Android pytorch_android_lite 1.10.0
* Pytorch_android_torchvision_lite 1.10.0
* TensorFlow Java 2.0
* Android Studio 4.0.1 or later

1. Prepare the external library OpenCV-Java

    1.1 Download OpenCV Android SDK
    Firstly, you need to download the OpenCV SDK for Android from the OpenCV [official website].(https://opencv.org/releases/) In our project, we use version 4.5.4.
   
    1.2 Import OpenCV library into Android Studio project
    Then, unzip the downloaded OpenCV Android SDK file. Then open Android Studio, select `"File" ->"New" ->"Import Module"`. Afterwards, select the Java folder in the decompressed OpenCV Android SDK directory and click "Finish" to import.

    1.3 Configure project dependencies
   Add dependencies to the OpenCV library in the build.gradle file of the project:

    ```
    dependencies {
    implementation project(':opencv')
    }
    ```
    Add dependencies to the OpenCV library in the build.gradle file of the :app:
    ```
    implementation project(path: ':OpenQCV')
    ```
    
2. Build with Android Studio
    Start Android Studio, then open the project located in `WOAD-Moblie-ObjectDetection`. Note the app's `build.gradle` file has the following lines:

    ```
    implementation 'org.pytorch:pytorch_android_lite:1.10.0'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.10.0'
    implementation 'org.tensorflow:tensorflow-lite:2.4.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.4.0'
    ```
    
The installation time will take no longer than 60 minutes on a "normal" desktop computer and Android smart phone.

### Demo
Select an Android emulator or device to run the app. You can go through the included example test images to see the detection results.

If you have a WOAD hardware device, after enabling the hardware device client, you can select the SelectButton on the interface to start the phone as a server for streaming testing of the device.

The important code is located in the following positions:

```
WOAD-Moblie-ObjectDetection\app\src\main\java\org\pytorch\demo\objectdetection\MainActivity.java
WOAD-Moblie-ObjectDetection\app\src\main\java\org\tensorflow\lite\examples\detection\tflite\YoloV5Classifier.java
WOAD-Moblie-ObjectDetection\app\src\main\java\org\tensorflow\lite\examples\detection\tflite\YoloV5ClassifierDetect.java
```
