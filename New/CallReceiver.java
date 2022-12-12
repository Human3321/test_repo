package com.example.a1209_app;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Vibrator;
import android.support.annotation.RequiresApi;
import android.telephony.PhoneNumberUtils;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutionException;

public class CallReceiver extends BroadcastReceiver {

    String phonestate;
    public static final String TAG_phoneState = "PHONE STATE";
    String url = "http://13.124.192.194:54103/user/"; // 서버 IP 주소

    // 신고용 url
    String reportUrl = "http://13.124.192.194:54103/report/";
    // 신고용 전역 휴대폰번호
    String phoneNumtoReport;

    public GettingPHP gPHP;
    String result = "";

    @RequiresApi(api = Build.VERSION_CODES.O)
    @SuppressLint("MissingPermission")

    @Override
    public void onReceive(Context context, Intent intent) {

        if (intent.getAction().equals("android.intent.action.PHONE_STATE")) {

            //TelecomManager telephonyManager = (TelecomManager) context.getSystemService(Context.TELECOM_SERVICE);

            Bundle extras = intent.getExtras();

            if (extras != null) {

                // 현재 폰 상태 가져옴
                String state = extras.getString(TelephonyManager.EXTRA_STATE);

                // 중복 호출 방지
                if (state.equals(phonestate)) {
                    return;
                } else {
                    phonestate = state;
                }

                // [벨 울리는 중]
                if (state.equals(TelephonyManager.EXTRA_STATE_RINGING)) {

                    if(MainActivity.use_set == true) {
                        String phone;

                        if (intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER) != null) {
                            // 수신 번호 가져옴
                            phone = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER);

                            try {
                                // 서버에 수신 전화번호 보내서 결과 받아옴
                                gPHP = new GettingPHP();
                                result = gPHP.execute(url + phone).get();
                                Log.d("res_11", result);

                                if (result.length() >= 7) {
                                    String full = result;
                                    String split[] = full.split(":");
                                    String s = split[1];
                                    String s1[] = s.split("]");
                                    MainActivity.txt_cicd.setText("누적 신고 횟수 : {" + s1[0]);
                                    Toast.makeText(context, "주의! 신고 {" + s1[0] + "회 누적된 번호입니다.", Toast.LENGTH_LONG).show();
                                } else {
                                    Toast.makeText(context, "깨끗", Toast.LENGTH_LONG).show();
                                }

                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }
                }
                // [통화 중]
                else if (state.equals(TelephonyManager.EXTRA_STATE_OFFHOOK)) {
                    if(intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)!=null) {

                        // 소켓 통신 스레드
                        Thread t = new Thread(() -> {
                            Socket clientSocket = new Socket();
                            InetSocketAddress ipep = new InetSocketAddress(MainActivity.IP, MainActivity.Port);

                            try {
                                clientSocket.connect(ipep);
                            } catch (IOException e) {
                                e.printStackTrace();
                            }

                            // 소켓이 접속이 완료되면 inputstream과 outputstream을 받는다.
                            try (InputStream receiver = clientSocket.getInputStream();) {
                                byte[] datalength = new byte[4];
                                // 데이터 길이를 받는다.
                                receiver.read(datalength, 0, 4);

                                // ByteBuffer를 통해 little 엔디언 형식으로 데이터 길이를 구한다.
                                ByteBuffer b = ByteBuffer.wrap(datalength);
                                b.order(ByteOrder.LITTLE_ENDIAN);
                                int length = b.getInt();

                                // 데이터를 받을 버퍼를 선언한다.
                                byte [] data = new byte[length];
                                // 데이터를 받는다.
                                receiver.read(data, 0, length);

                                // byte형식의 데이터를 string형식으로 변환한다.
                                String msg = new String(data, "UTF-8");
                                // 스트링 변환 이후 int로 변환(= 최종 값)
                                int msg1 = parseInt(msg);

                                // 받아온 값 이용할 수 있도록 가공 완.
                                MainActivity.isVP = msg1;
                                System.out.println(msg1);
                            }
                            catch (Throwable e) {
                                e.printStackTrace();
                            }
                        });

                        t.start();

                        try {
                            t.join();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }

                        if(MainActivity.isVP == 1){
                            Toast.makeText(context, "보이스피싱 의심 전화입니다 !", Toast.LENGTH_SHORT).show();
                            if(MainActivity.vib_mode==true){
                                vibrator.vibrate(3000);
                            }
                        } else {

                        }
                    }
                }
                // [통화종료]
                else if (state.equals(TelephonyManager.EXTRA_STATE_IDLE)) {
	        if(intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)!=null) {
                        String phone = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER);
                        phoneNumtoReport = phone;

                        // 받아온 판별 결과가 1이라면 자동 신고
                        if (MainActivity.isVP == 1) {
                            // 서버에 수신 전화번호 신고
                            gPHP = new GettingPHP();
                            gPHP.execute(reportUrl+phoneNumtoReport);
                            Toast.makeText(context, "보이스피싱 주의! 서버에 자동 신고되었습니다.", Toast.LENGTH_LONG).show();
                        }
                    }

                }
            }
        }
    }

    class GettingPHP extends AsyncTask<String, String, String> {

        protected void onPreExecute(){
        }

        // php 에서 데이터 읽어옴
        @Override
        protected String doInBackground(String... params) { // params : 전화번호
            Log.d("1conn_1", "1 ok");

            StringBuilder jsonHtml = new StringBuilder();
            try {
                URL phpUrl = new URL(params[0]);
                HttpURLConnection conn = (HttpURLConnection) phpUrl.openConnection();
                Log.d("2conn_2", "2 ok");
                Log.d("conn_state", String.valueOf(conn));

                if (conn != null) {
                    //conn.setConnectTimeout(10000);
                    //conn.setUseCaches(false);
                    Log.d("3conn_3", "3 ok");
                    int con_state = conn.getResponseCode();
                    Log.d("conn_r_code", String.valueOf(con_state));

                    if (conn.getResponseCode() == HttpURLConnection.HTTP_OK) {
                        Log.d("conn_eccc", "연결 ok");
                        Log.d("c_url", String.valueOf(conn.getURL()));
                        BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));
                        while (true) {
                            String line = br.readLine();
                            jsonHtml.append(line);
                            publishProgress(br.readLine());

                            Log.d("line_value", line);
                            if (line == null)
                                break;
                        }
                        br.close();
                    }
                    conn.disconnect();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            return jsonHtml.toString();
        }


        // 가져온 데이터 활용
        /*@Override
        protected void onPostExecute(String str) {
            String full = str;
            String split[] = full.split(":");
            String s = split[1];
            String s1[] = s.split("]");

            Log.d("value_of_str", s1[0]);
            result = s1[0];
            Log.d("result 1", "1: "+result);
        }*/

    }

}
