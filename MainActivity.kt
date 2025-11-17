package com.example.cybersafeconnect

import android.Manifest
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.cybersafeconnect.ui.theme.CyberSafeConnectTheme

class MainActivity : ComponentActivity() {

    private lateinit var wifiReceiver: WifiReceiver

    companion object {
        private const val REQ_CODE_LOCATION = 101
        private const val REQ_CODE_NOTIF = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Request required runtime permissions
        requestLocationPermissionIfNeeded()
        requestNotificationPermissionIfNeeded()

        // Create and register the receiver dynamically (works on Android 10+)
        wifiReceiver = WifiReceiver()
        val filter = IntentFilter(ConnectivityManager.CONNECTIVITY_ACTION)
        registerReceiver(wifiReceiver, filter)

        setContent {
            CyberSafeConnectTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(text = "🛡️ CyberSafe Connect", style = MaterialTheme.typography.headlineMedium)
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(text = "Monitoring Wi-Fi networks and alerting for insecure connections.")
                        Spacer(modifier = Modifier.height(20.dp))
                        Text(text = "Make sure Location is ON and you granted Location & Notification permissions.")
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            unregisterReceiver(wifiReceiver)
        } catch (_: Exception) { /* ignore */ }
    }

    private fun requestLocationPermissionIfNeeded() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                REQ_CODE_LOCATION
            )
        }
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    REQ_CODE_NOTIF
                )
            }
        }
    }
}
