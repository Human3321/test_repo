package com.example.newsttexam_1209;

import static java.lang.Integer.parseInt;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    // 인텐트 선언
    Intent intent;

    // STT Recognizer 선언
    SpeechRecognizer speechRecognizer;

    // UI 요소 선언
    Button sttBtn;
    EditText textView;

    // 권한 여부 판단을 위한 변수 선언
    final int PERMISSION = 1;

    // 현재 녹음중인지 여부
    boolean recording = false;

    // Todo: 1212
    // IP 주소
    static String IP = "3.36.54.30";
    static int Port = 59971;
    Handler mHandler;
    Socket rec;
    DataOutputStream dos;
    DataInputStream dis;

    void connect(String msg) {

        Thread sendText = new Thread(new Runnable() {
            @Override
            public void run() {

                // 데이터 생성 부분.
                // string을 byte배열 형식으로 변환한다.
                byte[] data = msg.getBytes();
                // ByteBuffer를 통해 데이터 길이를 byte형식으로 변환한다.
                ByteBuffer b = ByteBuffer.allocate(4);

                // byte포멧은 little 엔디언이다.
                b.order(ByteOrder.LITTLE_ENDIAN);
                b.putInt(data.length);

                try{
                    rec = new Socket(IP, Port);
                } catch (IOException e){
                    Log.d("server", e.toString());
                }

                try {
                    dos = new DataOutputStream(rec.getOutputStream());
                    dis = new DataInputStream(rec.getInputStream());
                } catch (IOException e){
                    Log.d("stream", e.toString());
                }

                try {
                    // 데이터 길이 전송
                    dos.write(b.array(), 0, 4);
                    // 데이터 전송
                    dos.write(data);

                    data = new byte[4];
                    // 데이터 길이를 받는다.
                    dis.read(data, 0, 4);

                    // ByteBuffer를 통해 little 엔디언 형식으로 데이터 길이를 구한다.
                    b = ByteBuffer.wrap(data);
                    b.order(ByteOrder.LITTLE_ENDIAN);
                    int length = b.getInt();

                    // 데이터를 받을 버퍼를 선언한다.
                    data = new byte[length];
                    // 데이터를 받는다.
                    dis.read(data, 0, length);

                    // byte형식의 데이터를 string형식으로 변환한다.
                    String ms = new String(data, "UTF-8");
                    // 스트링 변환 이후 int로 변환
                    int msg1 = parseInt(ms);

                    System.out.println(msg1);

                } catch (Exception e) {
                    Log.d("sender error", e.toString());
                }
            }
        });

        sendText.start();
        try {
            sendText.join();
        } catch (InterruptedException e) {

        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 빌드 버전에 따라 권한 요청
        CheckPermission();

        // UI 설정
        textView = findViewById(R.id.contentsTextView);
        sttBtn = findViewById(R.id.sttStart);

        // 인텐트 생성: 음성녹음 - RecognizerIntent 객체 생성
        intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE,getPackageName());
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE,"ko-KR");

        // 버튼 클릭 동작 리스너
        sttBtn.setOnClickListener(v -> {
            if (!recording) {   //녹음 시작
                StartRecord();
                Toast.makeText(getApplicationContext(), "지금부터 음성으로 기록합니다.", Toast.LENGTH_SHORT).show();
            }
            else {  //이미 녹음 중이면 녹음 중지
                StopRecord();
            }
        });
    }

    // 녹음 시작
    void StartRecord() {
        recording = true;

        // UX
        sttBtn.setText("텍스트 변환 중지");

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(getApplicationContext());
        speechRecognizer.setRecognitionListener(listener);
        // 녹음 시작
        speechRecognizer.startListening(intent);
    }

    // 녹음 중지
    void StopRecord() {
        recording = false;

        // UX
        sttBtn.setText("텍스트 변환 시작");

        // 녹음 중지
        speechRecognizer.stopListening();
        Toast.makeText(getApplicationContext(), "음성 기록을 중지합니다.", Toast.LENGTH_SHORT).show();
    }

    // 퍼미션 체크
    void CheckPermission() {
        // 안드로이드 버전이 6.0 이상인 경우
        if ( Build.VERSION.SDK_INT >= 23 ){
            // 인터넷이나 녹음 권한이 없으면 권한 요청
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.INTERNET) == PackageManager.PERMISSION_DENIED
             || ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_DENIED ) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.INTERNET,
                                                               Manifest.permission.RECORD_AUDIO},PERMISSION);
            }
        }
    }

    // 리스너: 사용자가 말을 멈추면 onEndOfSpeech() 호출, 에러가 발생했다면 onError(), 음성인식 결과가 제대로 나왔다면 onResults() 호출
    private RecognitionListener listener = new RecognitionListener() {
        @Override
        public void onReadyForSpeech(Bundle params) {

        }

        @Override
        public void onBeginningOfSpeech() {

        }

        @Override
        public void onRmsChanged(float rmsdB) {

        }

        @Override
        public void onBufferReceived(byte[] buffer) {

        }

        @Override
        public void onEndOfSpeech() {

        }

        @Override
        public void onError(int error) {
            String message;

            switch (error) {
                case SpeechRecognizer.ERROR_AUDIO:
                    message = "오디오 오류";
                    break;
                case SpeechRecognizer.ERROR_CLIENT:
                    //message = "클라이언트 에러";
                    //speechRecognizer.stopListening()을 호출하면 발생하는 에러
                    return; //토스트 메세지 출력 X
                case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS:
                    message = "권한이 없습니다.";
                    break;
                case SpeechRecognizer.ERROR_NETWORK:
                    message = "네트워크 에러";
                    break;
                case SpeechRecognizer.ERROR_NETWORK_TIMEOUT:
                    message = "네트워크 타임아웃";
                    break;
                case SpeechRecognizer.ERROR_NO_MATCH:
                    //message = "찾을 수 없음";
                    //녹음을 오래하거나 speechRecognizer.stopListening()을 호출하면 발생하는 에러
                    //speechRecognizer 다시 생성해 녹음 재개
                    if (recording)
                        StartRecord();
                    return; //토스트 메세지 출력 X
                case SpeechRecognizer.ERROR_RECOGNIZER_BUSY:
                    message = "RECOGNIZER가 바쁨";
                    break;
                case SpeechRecognizer.ERROR_SERVER:
                    message = "서버 오류";
                    break;
                case SpeechRecognizer.ERROR_SPEECH_TIMEOUT:
                    message = "대기시간 초과";
                    break;
                default:
                    message = "알 수 없는 오류 발생";
                    break;
            }
            Toast.makeText(getApplicationContext(), "에러가 발생하였습니다. : " + message,Toast.LENGTH_SHORT).show();
        }

        // 인식 결과가 준비됐을 때 호출
        // 기존 text 에 인식결과를 이어붙인 text 출력
        @Override
        public void onResults(Bundle results) {
            // 말을 하면 ArrayList 에 단어를 넣고 textView 에 단어를 이어줍니다.
            ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
            String originText = textView.getText().toString();  //기존 text

            String newText = "";
            for(int i = 0; i < matches.size() ; i++){
                // 여기서 텍스트뷰에 표시랑 동시에 보내주면?
                newText += matches.get(i);
            }

            textView.setText(originText + newText + " ");	//기존의 text에 인식 결과를 이어붙임
            String finalNewText = textView.getText().toString();

            // Todo: 적절한 길이 찾기. 250은 너무 긴 것 같아서 일단 줄여둠
            // 텍스트 길이가 100 이상이 되면
            if (textView.length() > 150) {
                // 소켓으로 전송
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        send(finalNewText);
                    }
                }).start();

                // 텍스트 초기화(이미 보냈으니)
                textView.setText("");
                Toast.makeText(getApplicationContext(), "서버로 텍스트를 전송했습니다.", Toast.LENGTH_SHORT).show();
            }
            // 녹음버튼을 누를 때까지 계속 녹음해야 하므로 녹음 재개
            speechRecognizer.startListening(intent);
        }

        @Override
        public void onPartialResults(Bundle partialResults) {

        }

        @Override
        public void onEvent(int eventType, Bundle params) {

        }
    };

    // 전송 메서드
    public void send(String msg) {
        // 소켓을 선언한다.
        try (Socket client = new Socket()) {
            // 소켓에 접속하기 위한 접속 정보를 선언한다.
            InetSocketAddress ipep = new InetSocketAddress(IP, Port);
            // 소켓 접속!
            client.connect(ipep);

            // 소켓이 접속이 완료되면 inputstream과 outputstream을 받는다.
            try (OutputStream sender = client.getOutputStream();
                 InputStream receiver = client.getInputStream();) {

                // string을 byte배열 형식으로 변환한다.
                byte[] data = msg.getBytes();
                // ByteBuffer를 통해 데이터 길이를 byte형식으로 변환한다.
                ByteBuffer b = ByteBuffer.allocate(4);

                // byte포멧은 little 엔디언이다.
                b.order(ByteOrder.LITTLE_ENDIAN);
                b.putInt(data.length);

                // 데이터 길이 전송
                sender.write(b.array(), 0, 4);
                // 데이터 전송
                sender.write(data);
                Toast.makeText(getApplicationContext(), "데이터 전송함",Toast.LENGTH_SHORT).show();

                data = new byte[4];
                // 데이터 길이를 받는다.
                receiver.read(data, 0, 4);

                // ByteBuffer를 통해 little 엔디언 형식으로 데이터 길이를 구한다.
                b = ByteBuffer.wrap(data);
                b.order(ByteOrder.LITTLE_ENDIAN);
                int length = b.getInt();

                // 데이터를 받을 버퍼를 선언한다.
                data = new byte[length];
                // 데이터를 받는다.
                receiver.read(data, 0, length);

                // byte형식의 데이터를 string형식으로 변환한다.
                msg = new String(data, "UTF-8");
                // 스트링 변환 이후 int로 변환
                int msg1 = parseInt(msg);

                Toast.makeText(getApplicationContext(), "데이터 받음: " + msg1,Toast.LENGTH_SHORT).show();

                // 콘솔에 출력한다.
                System.out.println(msg);
            }
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }
}