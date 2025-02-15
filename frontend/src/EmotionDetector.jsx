import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Camera, Loader2, Power, PowerOff, Smile, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const EmotionDetector = () => {
  const [emotion, setEmotion] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Enhanced emotion mapping with more descriptive labels
  const emotionMap = {
    0: {
      label: "Anger",
      emoji: "😠",
      description: "Feeling frustrated or irritated",
    },
    1: {
      label: "Disgust",
      emoji: "🤢",
      description: "Feeling repulsed or revolted",
    },
    2: {
      label: "Fear",
      emoji: "😱",
      description: "Experiencing anxiety or dread",
    },
    3: {
      label: "Happiness",
      emoji: "😄",
      description: "Feeling joyful and content",
    },
    4: {
      label: "Sadness",
      emoji: "😢",
      description: "Feeling down or melancholic",
    },
    5: {
      label: "Surprise",
      emoji: "😮",
      description: "Caught off guard or amazed",
    },
    6: {
      label: "Neutral",
      emoji: "😐",
      description: "Feeling calm and balanced",
    },
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setError(null);
    } catch (error) {
      console.error("Error accessing camera:", error);
      setError("Unable to access camera. Please check your permissions.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const toggleCamera = () => {
    if (isCameraOn) {
      stopCamera();
    } else {
      startCamera();
    }
    setIsCameraOn(!isCameraOn);
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext("2d");
      context.drawImage(videoRef.current, 0, 0, 400, 300);
      const imageDataUrl = canvasRef.current.toDataURL("image/jpeg");
      setCapturedImage(imageDataUrl);
      setEmotion(null);
      detectEmotion(imageDataUrl);
    }
  };

  const detectEmotion = async (imageData) => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await axios.post(
        "http://localhost:5000/detect-emotion",
        { image: imageData },
        { headers: { "Content-Type": "application/json" } }
      );

      const detectedEmotion = response.data.emotion;
      setEmotion(emotionMap[detectedEmotion]);
    } catch (error) {
      console.error("Emotion detection error:", error);
      setError("Error detecting emotion. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-50 to-blue-100">
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Smile className="w-8 h-8 text-blue-600" />
            <h1 className="text-2xl font-bold text-gray-800">
              Emotion Detector
            </h1>
          </div>
          <Button
            variant={isCameraOn ? "destructive" : "default"}
            onClick={toggleCamera}
            className="flex items-center space-x-2"
          >
            {isCameraOn ? (
              <PowerOff className="mr-2" />
            ) : (
              <Power className="mr-2" />
            )}
            {isCameraOn ? "Turn Off Camera" : "Turn On Camera"}
          </Button>
        </div>
      </header>

      <main className="flex-grow container mx-auto px-4 py-8">
        <Card className="w-full max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle className="text-center flex justify-center items-center space-x-2">
              <Camera className="w-6 h-6 text-blue-600" />
              <span>Live Emotion Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Camera Feed */}
            <div className="relative mb-6 rounded-lg overflow-hidden shadow-md">
              {isCameraOn ? (
                <video
                  ref={videoRef}
                  autoPlay
                  className="w-full aspect-video object-cover"
                />
              ) : (
                <div className="w-full aspect-video bg-gray-200 flex items-center justify-center">
                  <p className="text-gray-500">Camera is turned off</p>
                </div>
              )}
              <canvas
                ref={canvasRef}
                width={400}
                height={300}
                className="hidden"
              />
            </div>

            {/* Capture Button */}
            <Button
              onClick={captureImage}
              disabled={!isCameraOn || isLoading}
              className="w-full"
            >
              {isLoading ? (
                <Loader2 className="mr-2 animate-spin" />
              ) : (
                <Camera className="mr-2" />
              )}
              {isLoading ? "Detecting Emotion..." : "Capture Emotion"}
            </Button>

            {/* Error Handling */}
            {error && (
              <Alert variant="destructive" className="mt-4">
                <Info className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Emotion Result */}
            {emotion && (
              <div className="mt-6 p-6 bg-blue-50 border border-blue-200 rounded-lg text-center space-y-4">
                <div className="text-6xl">{emotion.emoji}</div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-800">
                    {emotion.label} Emotion
                  </h2>
                  <p className="text-gray-600">{emotion.description}</p>
                </div>
              </div>
            )}

            {/* Captured Image Preview */}
            {capturedImage && (
              <div className="mt-6 space-y-4">
                <h3 className="text-lg font-semibold text-center">
                  Captured Image
                </h3>
                <img
                  src={capturedImage}
                  alt="Captured"
                  className="w-full rounded-lg shadow-md"
                />
              </div>
            )}
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-4">
        <div className="container mx-auto text-center">
          <p className="flex items-center justify-center space-x-2">
            <span>© {new Date().getFullYear()} Emotion Detector</span>
            <Badge variant="secondary">AI Powered</Badge>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default EmotionDetector;
