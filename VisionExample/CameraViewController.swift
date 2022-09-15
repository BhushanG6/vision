//
//  Copyright (c) 2018 Google Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import AVFoundation
import CoreVideo
import MLImage
import MLKit
import AVKit
import UIKit
import Metal
import MetalKit
import AVFoundation
import SceneKit
import RealityKit
import ARKit
import VideoToolbox

var images:CVPixelBuffer?
var angArray=[0:true,30:true,60:true,90:true,120:true,150:true,180:true,210:true,240:true,270:true,300:true,330:true,360:true]
var bvalue=true
var booleanvalue=false
var eventfront=true
var eventleft=false
var e1=true
var r:Double=0
var ang:Double?
var theta:Double?
var disp:Double=0


var mid:simd_float3=[0.0,0.0,0.0]

var leftshld1:simd_float3=[0.0,0.0,0.0]
var rightshld1:simd_float3=[0.0,0.0,0.0]

var leftshld2:simd_float3=[0.0,0.0,0.0]
var rightshld2:simd_float3=[0.0,0.0,0.0]

var j:Int=0
var prevFrame:CGRect?
@objc(CameraViewController)
class CameraViewController: UIViewController,ARSessionDelegate {
    var headJointPosx : Float = 0.0
    var headJointPosy : Float = 0.0
    var headJointPosz : Float = 0.0
    var arrayofallcenter = [simd_float3]()
    var center : simd_float3 = [0,0,0]
    var player: AVAudioPlayer?
    var event = true
    private let confidenceControl = UISegmentedControl(items: ["Low", "Medium", "High"])
    private let rgbRadiusSlider = UISlider()
    
    private let session = ARSession()
    @IBOutlet weak var saveButton: UIButton!
    private var renderer: Renderer!
    // MARK: - Properties
    var trackingStatus: String = ""
    
    var start = true
    var countframeimage = 0
  private let detectors: [Detector] = [
     // .onDeviceFace,
//    .onDeviceText,
//    .onDeviceTextChinese,
//    .onDeviceTextDevanagari,
//    .onDeviceTextJapanese,
//    .onDeviceTextKorean,
//    .onDeviceBarcode,
//    .onDeviceImageLabel,
//    .onDeviceImageLabelsCustom,
//    .onDeviceObjectProminentNoClassifier,
    .onDeviceObjectProminentWithClassifier,
//    .onDeviceObjectMultipleNoClassifier,
//    .onDeviceObjectMultipleWithClassifier,
//    .onDeviceObjectCustomProminentNoClassifier,
//    .onDeviceObjectCustomProminentWithClassifier,
//    .onDeviceObjectCustomMultipleNoClassifier,
//    .onDeviceObjectCustomMultipleWithClassifier,
    .pose,
//    .poseAccurate,
//    .segmentationSelfie,
  ]
  @IBOutlet var sceneView: ARSCNView!
  private var currentDetector: Detector = .pose
  private var isUsingFrontCamera = false
  private var previewLayer: AVCaptureVideoPreviewLayer!
  private lazy var captureSession = AVCaptureSession()
  private lazy var sessionQueue = DispatchQueue(label: Constant.sessionQueueLabel)
  private var lastFrame: CMSampleBuffer?

  private lazy var previewOverlayView: UIImageView = {

    precondition(isViewLoaded)
    let previewOverlayView = UIImageView(frame: .zero)
    previewOverlayView.contentMode = UIView.ContentMode.scaleAspectFill
    previewOverlayView.translatesAutoresizingMaskIntoConstraints = false
    return previewOverlayView
  }()

  private lazy var annotationOverlayView: UIView = {
    precondition(isViewLoaded)
    let annotationOverlayView = UIView(frame: .zero)
    annotationOverlayView.translatesAutoresizingMaskIntoConstraints = false
    return annotationOverlayView
  }()

  /// Initialized when one of the pose detector rows are chosen. Reset to `nil` when neither are.
  private var poseDetector: PoseDetector? = nil

  /// Initialized when a segmentation row is chosen. Reset to `nil` otherwise.
  private var segmenter: Segmenter? = nil

  /// The detector mode with which detection was most recently run. Only used on the video output
  /// queue. Useful for inferring when to reset detector instances which use a conventional
  /// lifecyle paradigm.
  private var lastDetector: Detector?

  // MARK: - IBOutlets

  @IBOutlet private weak var cameraView: UIView!
    //let configuration = ARBodyTrackingConfiguration()

    @IBOutlet weak var showAngle: UILabel!
    // MARK: - UIViewController
    func createSpinnerView() {
        let child = SpinnerViewController()

        // add the spinner view controller
        addChild(child)
        child.view.frame = view.frame
        view.addSubview(child.view)
        child.didMove(toParent: self)

        // wait two seconds to simulate some work happening
//        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
//            // then remove the spinner view controller
//            child.willMove(toParent: nil)
//            child.view.removeFromSuperview()
//            child.removeFromParent()
//        }
    }
  override func viewDidLoad() {
    super.viewDidLoad()
    print("Entered into renderer")
    session.delegate = self

    
      guard let device = MTLCreateSystemDefaultDevice() else {
          print("Metal is not supported on this device")
          return
    }
      if device != nil
      {
          print("MyDevice",device.makeDefaultLibrary())

      }
    //self.showAngle.text="Person is stable"
      if let view = view as? MTKView {
          view.device = device
          
          view.backgroundColor = UIColor.clear
          // we need this to enable depth test
          view.depthStencilPixelFormat = .depth32Float
          view.contentScaleFactor = 1
          view.delegate = self
          
          // Configure the renderer to draw to the view
          renderer = Renderer(session: session, metalDevice: device, renderDestination: view)
          renderer.drawRectResized(size: view.bounds.size)
          
      }
      renderer.fireTimer()
      renderer.createderectory()

    //  previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
  //    setUpPreviewOverlayView()
  //    setUpAnnotationOverlayView()
//      setUpCaptureSessionOutput()
//      setUpCaptureSessionInput()
  }
    override func viewWillAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        //super.viewDidAppear(animated)

        //startSession()
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = .sceneDepth

        // Run the view's session
        session.run(configuration)
        
        // The screen shouldn't dim during AR experiences.
        UIApplication.shared.isIdleTimerDisabled = true
    }
    override func viewWillDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        stopSession()

        // Pause the view's session
        session.pause()
    }
    @IBAction func showPointCloud(_ sender: Any) {
        let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
        let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "ShowPointCloudViewController") as! ShowPointCloudViewController
           
        secondViewController.filename = Helper().retrievePathnameFromKeychain() ?? ""
        self.navigationController?.pushViewController(secondViewController, animated: true)
    }

    func playSound(soundname : String) {
        guard let url = Bundle.main.url(forResource: soundname, withExtension: "mp3") else { return }

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            /* The following line is required for the player to work on iOS 11. Change the file type accordingly*/
            player = try AVAudioPlayer(contentsOf: url, fileTypeHint: AVFileType.mp3.rawValue)

            /* iOS 10 and earlier require the following line:
            player = try AVAudioPlayer(contentsOf: url, fileTypeHint: AVFileTypeMPEGLayer3) */

            guard let player = player else { return }

            player.play()

        } catch let error {
            print(error.localizedDescription)
        }
    }
  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()

    //previewLayer.frame = cameraView.frame
  }

  // MARK: - IBActions

  @IBAction func selectDetector(_ sender: Any) {
    presentDetectorsAlertController()
  }

  @IBAction func switchCamera(_ sender: Any) {
    isUsingFrontCamera = !isUsingFrontCamera
    removeDetectionAnnotations()
    setUpCaptureSessionInput()
  }

  // MARK: On-Device Detections
  private func detectFacesOnDevice(in image: VisionImage, width: CGFloat, height: CGFloat) {
    // When performing latency tests to determine ideal detection settings, run the app in 'release'
    // mode to get accurate performance metrics.
    let options = FaceDetectorOptions()
    options.landmarkMode = .none
    options.contourMode = .all
    options.classificationMode = .none
    options.performanceMode = .fast
    let faceDetector = FaceDetector.faceDetector(options: options)
    var faces: [Face] = []
    var detectionError: Error?
    do {
      faces = try faceDetector.results(in: image)
    } catch let error {
      detectionError = error
    }
    weak var weakSelf = self
    DispatchQueue.main.sync {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.updatePreviewOverlayViewWithLastFrame()
      if let detectionError = detectionError {
        print("Failed to detect faces with error: \(detectionError.localizedDescription).")
        return
      }
      guard !faces.isEmpty else {
        print("On-Device face detector returned no results.")
        return
      }

      for face in faces {
        let normalizedRect = CGRect(
          x: face.frame.origin.x / width,
          y: face.frame.origin.y / height,
          width: face.frame.size.width / width,
          height: face.frame.size.height / height
        )
        let standardizedRect = strongSelf.previewLayer.layerRectConverted(
          fromMetadataOutputRect: normalizedRect
        ).standardized
        UIUtilities.addRectangle(
          standardizedRect,
          to: strongSelf.annotationOverlayView,
          color: UIColor.green
        )
        strongSelf.addContours(for: face, width: width, height: height)
      }
    }
  }

  private func detectPose(in image: MLImage, width: CGFloat, height: CGFloat) {
    if let poseDetector = self.poseDetector {
      var poses: [Pose] = []
      var detectionError: Error?
        print("Inside poseDetector1")

      do {
          print("Hii pose")
          if image != nil
          {
              print("MyImage",image)

          }
        poses = try poseDetector.results(in: image)
      } catch let error {
          print("Inside Error")
        detectionError = error
      }
      weak var weakSelf = self

      DispatchQueue.main.sync {
        guard let strongSelf = weakSelf else {
          print("Self is nil!")
          return
        }
        strongSelf.updatePreviewOverlayViewWithLastFrame()
        if let detectionError = detectionError {
          print("Failed to detect poses with error: \(detectionError.localizedDescription).")
          return
        }
        guard !poses.isEmpty else {
          print("Pose detector returned no results.")
          return
        }
        print("Inside poseDetector2")
        // Pose detected. Currently, only single person detection is supported.
        poses.forEach { pose in
//          let poseOverlayView = UIUtilities.createPoseOverlayView(
//            forPose: pose,
//            inViewWithBounds: strongSelf.annotationOverlayView.bounds,
//            lineWidth: Constant.lineWidth,
//            dotRadius: Constant.smallDotRadius,
//            positionTransformationClosure: { (position) -> CGPoint in
//              return strongSelf.normalizedPoint(
//                fromVisionPoint: position, width: width, height: height)
//            }
//          )
            let leftarm=pose.landmark(ofType: .leftShoulder).position
            let rightarm=pose.landmark(ofType: .rightShoulder).position
            if e1{
               
            }
            if eventfront {

                DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                leftshld1 = [Float(leftarm.x), Float(leftarm.y), Float(leftarm.z)]
                rightshld1 = [Float(rightarm.x), Float(rightarm.y), Float(rightarm.z)]
                mid=(leftshld1+rightshld1)/2
               // print("turn")
//                print("-----------------------------------------------")
//                    print("Left Shoulder 1 is ", leftshld1)
//
//                    print("Right Shoulder 1 is ", rightshld1)

                                       
                e1=false
                    eventfront = false

                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 8) {
                   
                    eventleft = true
                }
            }
            if eventleft {

               // DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                leftshld2 = [Float(leftarm.x), Float(leftarm.y), Float(leftarm.z)]
                rightshld2 = [Float(rightarm.x), Float(rightarm.y), Float(rightarm.z)]
                
//                    leftshld2=[ (leftshld2.x * cos(90 * Float(Double.pi) / 180)) + (leftshld2.z * sin(90 * Float(Double.pi) / 180)) , leftshld2.y, (leftshld2.z * cos(90 * Float(Double.pi) / 180)) - (leftshld2.x * sin(90 * Float(Double.pi) / 180))]
              //  print("***********************************************")

//                    print("Left Shoulder 2 is ", leftshld2)
//
//                    print("Right Shoulder 2 is ", rightshld2)
                
               // eventleft = false
                r = Double(self.distance3D(vector1: leftshld1, vector2: rightshld1))/2
                disp = Double(self.distance3D(vector1: leftshld1, vector2: leftshld2))
                var v1=leftshld1-mid
                var v2=leftshld2-mid
                //ang=asin(disp/(2*r))*2
                var m=sqrt(pow(v1.x,2)+pow(v1.y,2)+pow(v1.z,2))*(sqrt(pow(v2.x,2)+pow(v2.y,2)+pow(v2.z,2)))
                var dot1=dot(v1,v2)/m
               ang=acos(Double(dot1))
               theta = ang! * (180/3.14159)
//                    print("Radius is ", r)
//
//                     print("Displacement is ", disp)
                if theta! < 180 && bvalue{

                    if theta! > 170{
                        bvalue=false

                    }

                                        

                                    }
                if !bvalue{
                                        theta = 360 - theta!
                    
                    }
//                showAngle.layer.cornerCurve
                
                showAngle.text=String(theta!)
                print("Angle Printed  =======", theta!)

                                    

                                    

                                    

                                    switch theta!{

                                    

                                    case 0..<10:

                                        if angArray[0]==true

                                        {

                                            angArray[0]=false

                                            return textToSpeech(Number: 0)

                                        }

                                        

                                    case 28..<35:

                                        

                                        if angArray[30]==true

                                        {

                                            angArray[30]=false

                                            return textToSpeech(Number: 30)

                                        }

                                    case 55..<63:

                                        

                                        if angArray[60]==true

                                        {

                                            angArray[60]=false

                                            return textToSpeech(Number: 60)

                                        }

                                    case 85..<92:

                                        if angArray[90]==true

                                        {

                                            angArray[90]=false

                                            return textToSpeech(Number: 90)

                                        }

                                    case 115..<123:


                                        if angArray[120]==true

                                        {

                                            angArray[120]=false

                                            return textToSpeech(Number: 120)

                                        }

                                    case 145..<153:


                                        if angArray[150]==true

                                        {

                                            angArray[150]=false

                                            return textToSpeech(Number: 150)

                                        }

                                    case 170..<185:

                                        if angArray[180]==true

                                        {

                                            angArray[180]=false

                                            return textToSpeech(Number: 180)

                                        }

                                    case 205..<213:


                                        if angArray[210]==true

                                        {

                                            angArray[210]=false

                                            return textToSpeech(Number: 210)

                                        }

                                    case 235..<243:


                                        if angArray[240]==true

                                        {

                                            angArray[240]=false

                                            return textToSpeech(Number: 240)

                                        }

                                    case 265..<273:


                                        if angArray[270]==true

                                        {

                                            angArray[270]=false

                                            return textToSpeech(Number: 270)

                                        }

                                    case 295..<303:


                                        if angArray[300]==true

                                        {

                                            angArray[300]=false

                                            return textToSpeech(Number: 300)

                                        }

                                    case 325..<333:

                                        if angArray[330]==true

                                        {

                                            angArray[330]=false

                                            return textToSpeech(Number: 330)

                                        }

                                    case 340..<360:


                                        if angArray[360]==true

                                        {

                                            angArray[360]=false

                                            return textToSpeech(Number: 360)

                                        }

                                    default:

                                        return

                                    }
                //}

               
            }
//            print("*********************************")
//            //print(pose.landmark(ofType: .RIGHT_SHOULDER))
//            print(pose.landmark(ofType: .leftEar).position)
//            print(pose.landmark(ofType: .rightEar).position)
//            print("________________________________________")
//            print(pose.landmark(ofType: .leftShoulder).position)
//            print(pose.landmark(ofType: .rightShoulder).position)
          // strongSelf.annotationOverlayView.addSubview(poseOverlayView)
        }
      }
    }
  }
    func textToSpeech(Number: Int){

            let utterance = AVSpeechUtterance(string: "\(Number) Degrees")



            // Configure the utterance.

            utterance.rate = 0.57

            utterance.pitchMultiplier = 0.8

            utterance.postUtteranceDelay = 0.2

            utterance.volume = 0.8



            // Retrieve the British English voice.

    //        let voice = AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.siri_male_en-GB_compact")



            // Assign the voice to the utterance.

            utterance.voice = AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.siri_male_en-GB_compact")

            

            let synthesizer = AVSpeechSynthesizer()

            synthesizer.speak(utterance)

        }
    func texttospeech(Number: String){

            let utterance = AVSpeechUtterance(string: Number)



            // Configure the utterance.

            utterance.rate = 0.57

            utterance.pitchMultiplier = 0.8

            utterance.postUtteranceDelay = 0.2

            utterance.volume = 0.8



            // Retrieve the British English voice.

    //        let voice = AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.siri_male_en-GB_compact")



            // Assign the voice to the utterance.

            utterance.voice = AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.siri_male_en-GB_compact")

            

            let synthesizer = AVSpeechSynthesizer()

            synthesizer.speak(utterance)

        }
    func distance3D(vector1: simd_float3, vector2: simd_float3) -> Float
        {
            let x: Float = (vector1.x - vector2.x) * (vector1.x - vector2.x)
            let y: Float = (vector1.y - vector2.y) * (vector1.y - vector2.y)
            let z: Float = (vector1.z - vector2.z) * (vector1.z - vector2.z)

            let temp = x + y + z
            return Float(sqrtf(Float(temp)))
        }
    func keepRadius(){
        let r = Double(self.distance3D(vector1: leftshld1, vector2: rightshld1))/2
    }


  private func detectSegmentationMask(in image: VisionImage, sampleBuffer: CMSampleBuffer) {
    guard let segmenter = self.segmenter else {
      return
    }
    var mask: SegmentationMask? = nil
    var segmentationError: Error?
    do {
      mask = try segmenter.results(in: image)
    } catch let error {
      segmentationError = error
    }
    weak var weakSelf = self
    DispatchQueue.main.sync {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.removeDetectionAnnotations()

      if let segmentationError = segmentationError {
        print(
          "Failed to perform segmentation with error: \(segmentationError.localizedDescription).")
        return
      }
      guard let mask = mask else {
        print("Segmenter returned empty mask.")
        return
      }
      guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        print("Failed to get image buffer from sample buffer.")
        return
      }

      UIUtilities.applySegmentationMask(
        mask: mask, to: imageBuffer,
        backgroundColor: UIColor.purple.withAlphaComponent(Constant.segmentationMaskAlpha),
        foregroundColor: nil)
      strongSelf.updatePreviewOverlayViewWithImageBuffer(imageBuffer)
    }
  }
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
/*
        let speare = SCNSphere(radius: 0.05)



        let headNode = SCNNode()
//        let toesNode = SCNNode()

//                    headNode.position = SCNVector3(0 ,  headJointPosy, 0)
//        headNode.geometry = speare

        let materials = SCNMaterial()
        materials.diffuse.contents = UIColor.red

        speare.materials = [materials]

        headNode.position = SCNVector3(x: headJointPosx, y: headJointPosy    , z: 0)
        headNode.geometry = speare
        node.addChildNode(headNode) */
//                }
//
//            }
        
//                    toesNode.position = SCNVector3(to)
        
        
        
        
        
        
    
}
  @objc func repeatFunc()
  {
      booleanvalue=true
  }
  private func detectObjectsOnDevice(
    in image: VisionImage,
    width: CGFloat,
    height: CGFloat,
    options: CommonObjectDetectorOptions
  ) {

    let detector = ObjectDetector.objectDetector(options: options)
    var objects: [Object] = []
    var detectionError: Error? = nil
    do {
      objects = try detector.results(in: image)
    } catch let error {
      detectionError = error
    }
    //var timer = Timer()
    
    weak var weakSelf = self
    DispatchQueue.main.sync {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.self.updatePreviewOverlayViewWithLastFrame()
      if let detectionError = detectionError {
        print("Failed to detect objects with error: \(detectionError.localizedDescription).")
        return

      }
      guard !objects.isEmpty else {
        print("On-Device object detector returned no results.")
        return
      }
      for object in objects {
        let normalizedRect = CGRect(
          x: object.frame.origin.x / width,
          y: object.frame.origin.y / height,
          width: object.frame.size.width / width,
          height: object.frame.size.height / height
        )
        let standardizedRect = strongSelf.previewLayer.layerRectConverted(
          fromMetadataOutputRect: normalizedRect
        ).standardized
        UIUtilities.addRectangle(
          standardizedRect,
          to: strongSelf.annotationOverlayView,
          color: UIColor.green
        )
        if j==0
        {
            prevFrame=object.frame
            j+=1
        }
        let timer = Timer.scheduledTimer(timeInterval: 0.5, target: self, selector: #selector(repeatFunc), userInfo: nil, repeats: true)

          if booleanvalue {
              if object.frame.minX != prevFrame!.minX && (abs(object.frame.minX-prevFrame!.minX)>7) || (abs(object.frame.minY-prevFrame!.minY)>7)
              {
                print("Person has moved")
                showAngle.text="Person has moved"
                  //DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                      self.texttospeech(Number:"Person has moved")
               // }

                  
              }
              else
              {
                  print("Person is stable")
//                  DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                      self.showAngle.text="Person is stable"
//                  }

              }
              if abs(object.frame.height-(prevFrame!.height))>8 || abs(object.frame.width-(prevFrame!.width))>8
              {
                 // showAngle.text="Frame size changed"
              }
              if prevFrame!.minX>object.frame.minX{
                  //showAngle.text="Person turned right"
              }
              else if(prevFrame!.minX<object.frame.minX)
              {
                 // showAngle.text="Person turned right"
              }
              else
              {
                  print("Front or back")
              }
              prevFrame=object.frame
              booleanvalue = false
          }
        //prevFrame=object.frame

        //var currentFrame:CGRect
        let label = UILabel(frame: standardizedRect)
        var description = ""
        
        if let trackingID = object.trackingID {
            
            print(object.frame)
            print(object.frame.minX,object.frame.minY,object.frame.maxX,object.frame.maxY,object.frame.midX,object.frame.midY,object.frame.origin,object.frame.height,object.frame.width,object.frame.size)
            
            description += "Object ID: " + trackingID.stringValue + "\n"
        }
        //description += object.frame+"\n"
        description += object.labels.enumerated().map { (index, label) in
          "Label \(index): \(label.text), \(label.confidence), \(label.index)"
        }.joined(separator: "\n")
        print("***********************")
        //label.text = description
        label.numberOfLines = 0
        label.adjustsFontSizeToFitWidth = true
        strongSelf.rotate(label, orientation: image.orientation)
        strongSelf.annotationOverlayView.addSubview(label)
      }
    }
  }

  // MARK: - Private

  private func setUpCaptureSessionOutput() {
    weak var weakSelf = self
    sessionQueue.async {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.captureSession.beginConfiguration()
      // When performing latency tests to determine ideal capture settings,
      // run the app in 'release' mode to get accurate performance metrics
      strongSelf.captureSession.sessionPreset = AVCaptureSession.Preset.medium

      let output = AVCaptureVideoDataOutput()
      output.videoSettings = [
        (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32BGRA
      ]
      output.alwaysDiscardsLateVideoFrames = true
      let outputQueue = DispatchQueue(label: Constant.videoDataOutputQueueLabel)
      output.setSampleBufferDelegate(strongSelf, queue: outputQueue)
      guard strongSelf.captureSession.canAddOutput(output) else {
        print("Failed to add capture session output.")
        return
      }
      strongSelf.captureSession.addOutput(output)
      strongSelf.captureSession.commitConfiguration()
    }
  }

  private func setUpCaptureSessionInput() {
    weak var weakSelf = self
    sessionQueue.async {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      let cameraPosition: AVCaptureDevice.Position = strongSelf.isUsingFrontCamera ? .front : .back
      guard let device = strongSelf.captureDevice(forPosition: cameraPosition) else {
        print("Failed to get capture device for camera position: \(cameraPosition)")
        return
      }
      do {
        strongSelf.captureSession.beginConfiguration()
        let currentInputs = strongSelf.captureSession.inputs
        for input in currentInputs {
          strongSelf.captureSession.removeInput(input)
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard strongSelf.captureSession.canAddInput(input) else {
          print("Failed to add capture session input.")
          return
        }
        strongSelf.captureSession.addInput(input)
        strongSelf.captureSession.commitConfiguration()
      } catch {
        print("Failed to create capture device input: \(error.localizedDescription)")
      }
    }
  }

  private func startSession() {
    weak var weakSelf = self
    sessionQueue.async {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.captureSession.startRunning()
    }
  }

  private func stopSession() {
    weak var weakSelf = self
    sessionQueue.async {
      guard let strongSelf = weakSelf else {
        print("Self is nil!")
        return
      }
      strongSelf.captureSession.stopRunning()
    }
  }

  private func setUpPreviewOverlayView() {
    cameraView.addSubview(previewOverlayView)
    NSLayoutConstraint.activate([
      previewOverlayView.centerXAnchor.constraint(equalTo: cameraView.centerXAnchor),
      previewOverlayView.centerYAnchor.constraint(equalTo: cameraView.centerYAnchor),
      previewOverlayView.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
      previewOverlayView.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),

    ])
  }
  
  private func setUpAnnotationOverlayView() {
    cameraView.addSubview(annotationOverlayView)
    NSLayoutConstraint.activate([
      annotationOverlayView.topAnchor.constraint(equalTo: cameraView.topAnchor),
      annotationOverlayView.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
      annotationOverlayView.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),
      annotationOverlayView.bottomAnchor.constraint(equalTo: cameraView.bottomAnchor),
    ])
  }

  private func captureDevice(forPosition position: AVCaptureDevice.Position) -> AVCaptureDevice? {
    if #available(iOS 10.0, *) {
      let discoverySession = AVCaptureDevice.DiscoverySession(
        deviceTypes: [.builtInWideAngleCamera],
        mediaType: .video,
        position: .unspecified
      )
      return discoverySession.devices.first { $0.position == position }
    }
    return nil
  }

  private func presentDetectorsAlertController() {
    let alertController = UIAlertController(
      title: Constant.alertControllerTitle,
      message: Constant.alertControllerMessage,
      preferredStyle: .alert
    )
    weak var weakSelf = self
    detectors.forEach { detectorType in
      let action = UIAlertAction(title: detectorType.rawValue, style: .default) {
        [unowned self] (action) in
        guard let value = action.title else { return }
        guard let detector = Detector(rawValue: value) else { return }
        guard let strongSelf = weakSelf else {
          print("Self is nil!")
          return
        }
        strongSelf.currentDetector = detector
        strongSelf.removeDetectionAnnotations()
      }
      if detectorType.rawValue == self.currentDetector.rawValue { action.isEnabled = false }
      alertController.addAction(action)
    }
    alertController.addAction(UIAlertAction(title: Constant.cancelActionTitleText, style: .cancel))
    present(alertController, animated: true)
  }

  private func removeDetectionAnnotations() {
    for annotationView in annotationOverlayView.subviews {
      annotationView.removeFromSuperview()
    }
  }

  private func updatePreviewOverlayViewWithLastFrame() {
    guard let lastFrame = lastFrame,
      let imageBuffer = CMSampleBufferGetImageBuffer(lastFrame)
    else {
      return
    }
    self.updatePreviewOverlayViewWithImageBuffer(imageBuffer)
    self.removeDetectionAnnotations()
  }

  private func updatePreviewOverlayViewWithImageBuffer(_ imageBuffer: CVImageBuffer?) {
    guard let imageBuffer = imageBuffer else {
      return
    }
    let orientation: UIImage.Orientation = isUsingFrontCamera ? .leftMirrored : .right
    let image = UIUtilities.createUIImage(from: imageBuffer, orientation: orientation)
    previewOverlayView.image = image
  }

  private func convertedPoints(
    from points: [NSValue]?,
    width: CGFloat,
    height: CGFloat
  ) -> [NSValue]? {
    return points?.map {
      let cgPointValue = $0.cgPointValue
      let normalizedPoint = CGPoint(x: cgPointValue.x / width, y: cgPointValue.y / height)
      let cgPoint = previewLayer.layerPointConverted(fromCaptureDevicePoint: normalizedPoint)
      let value = NSValue(cgPoint: cgPoint)
      return value
    }
  }

  private func normalizedPoint(
    fromVisionPoint point: VisionPoint,
    width: CGFloat,
    height: CGFloat
  ) -> CGPoint {
    let cgPoint = CGPoint(x: point.x, y: point.y)
    var normalizedPoint = CGPoint(x: cgPoint.x / width, y: cgPoint.y / height)
    normalizedPoint = previewLayer.layerPointConverted(fromCaptureDevicePoint: normalizedPoint)
    return normalizedPoint
  }

  private func addContours(for face: Face, width: CGFloat, height: CGFloat) {
    // Face
    if let faceContour = face.contour(ofType: .face) {
      for point in faceContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.blue,
          radius: Constant.smallDotRadius
        )
      }
    }

    // Eyebrows
    if let topLeftEyebrowContour = face.contour(ofType: .leftEyebrowTop) {
      for point in topLeftEyebrowContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.orange,
          radius: Constant.smallDotRadius
        )
      }
    }
    if let bottomLeftEyebrowContour = face.contour(ofType: .leftEyebrowBottom) {
      for point in bottomLeftEyebrowContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.orange,
          radius: Constant.smallDotRadius
        )
      }
    }
    if let topRightEyebrowContour = face.contour(ofType: .rightEyebrowTop) {
      for point in topRightEyebrowContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.orange,
          radius: Constant.smallDotRadius
        )
      }
    }
    if let bottomRightEyebrowContour = face.contour(ofType: .rightEyebrowBottom) {
      for point in bottomRightEyebrowContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.orange,
          radius: Constant.smallDotRadius
        )
      }
    }

    // Eyes
    if let leftEyeContour = face.contour(ofType: .leftEye) {
      for point in leftEyeContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.cyan,
          radius: Constant.smallDotRadius
        )
      }
    }
    if let rightEyeContour = face.contour(ofType: .rightEye) {
      for point in rightEyeContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.cyan,
          radius: Constant.smallDotRadius
        )
      }
    }

    // Lips
    if let topUpperLipContour = face.contour(ofType: .upperLipTop) {
      for point in topUpperLipContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.red,
          radius: Constant.smallDotRadius
        )
      }
    }
    if let bottomUpperLipContour = face.contour(ofType: .upperLipBottom) {
      for point in bottomUpperLipContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.red,
          radius: Constant.smallDotRadius
        )
      }
    }
    if let topLowerLipContour = face.contour(ofType: .lowerLipTop) {
      for point in topLowerLipContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.red,
          radius: Constant.smallDotRadius
        )
      }
    }
    if let bottomLowerLipContour = face.contour(ofType: .lowerLipBottom) {
      for point in bottomLowerLipContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.red,
          radius: Constant.smallDotRadius
        )
      }
    }

    // Nose
    if let noseBridgeContour = face.contour(ofType: .noseBridge) {
      for point in noseBridgeContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.yellow,
          radius: Constant.smallDotRadius
        )
      }
    }
    if let noseBottomContour = face.contour(ofType: .noseBottom) {
      for point in noseBottomContour.points {
        let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
        UIUtilities.addCircle(
          atPoint: cgPoint,
          to: annotationOverlayView,
          color: UIColor.yellow,
          radius: Constant.smallDotRadius
        )
      }
    }
  }

  /// Resets any detector instances which use a conventional lifecycle paradigm. This method is
  /// expected to be invoked on the AVCaptureOutput queue - the same queue on which detection is
  /// run.
  private func resetManagedLifecycleDetectors(activeDetector: Detector) {
//    if activeDetector == self.lastDetector {
//      // Same row as before, no need to reset any detectors.
//      return
//    }
    // Clear the old detector, if applicable.
//    switch self.lastDetector {
//    case .pose:
//      self.poseDetector = nil
//      break
////    case .segmentationSelfie:
////      self.segmenter = nil
////      break
//    default:
//      break
//    }
    // Initialize the new detector, if applicable.
    switch activeDetector {
    case .pose:
      // The `options.detectorMode` defaults to `.stream`
      let options = activeDetector == .pose ? PoseDetectorOptions() : AccuratePoseDetectorOptions()
      self.poseDetector = PoseDetector.poseDetector(options: options)
      break
//    case .segmentationSelfie:
//      // The `options.segmenterMode` defaults to `.stream`
//      let options = SelfieSegmenterOptions()
//      self.segmenter = Segmenter.segmenter(options: options)
//      break
    default:
      break
    }
    self.lastDetector = activeDetector
  }

  private func rotate(_ view: UIView, orientation: UIImage.Orientation) {
    var degree: CGFloat = 0.0
    switch orientation {
    case .up, .upMirrored:
      degree = 90.0
    case .rightMirrored, .left:
      degree = 180.0
    case .down, .downMirrored:
      degree = 270.0
    case .leftMirrored, .right:
      degree = 0.0
    }
    view.transform = CGAffineTransform.init(rotationAngle: degree * 3.141592654 / 180)
  }
}

// MARK: AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
      _ output: AVCaptureOutput,
      didOutput sampleBuffer: CMSampleBuffer,
      from connection: AVCaptureConnection
    ) {
        print("sample1",sampleBuffer)
      guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        print("Failed to get image buffer from sample buffer.")
        return
      }
        print("image",imageBuffer)
      // Evaluate `self.currentDetector` once to ensure consistency throughout this method since it
      // can be concurrently modified from the main thread.
      let activeDetector = self.currentDetector
      resetManagedLifecycleDetectors(activeDetector: activeDetector)
        
      lastFrame = sampleBuffer
      let visionImage = VisionImage(buffer: sampleBuffer)
      let orientation = UIUtilities.imageOrientation(
        fromDevicePosition: isUsingFrontCamera ? .front : .back
      )
      visionImage.orientation = orientation
      print("inside capture")
      guard let inputImage = MLImage(sampleBuffer: sampleBuffer) else {
        print("Failed to create MLImage from sample buffer.")
        return
      }
      inputImage.orientation = orientation

      let imageWidth = CGFloat(CVPixelBufferGetWidth(imageBuffer))
      let imageHeight = CGFloat(CVPixelBufferGetHeight(imageBuffer))
      var shouldEnableClassification = false
      var shouldEnableMultipleObjects = false
      switch activeDetector {
      case .onDeviceObjectProminentWithClassifier:
        shouldEnableClassification = true
      default:
        break
      }
  //    switch activeDetector {
  //    case .onDeviceObjectMultipleNoClassifier, .onDeviceObjectMultipleWithClassifier,
  //      .onDeviceObjectCustomMultipleNoClassifier, .onDeviceObjectCustomMultipleWithClassifier:
  //      shouldEnableMultipleObjects = true
  //    default:
  //      break
  //    }

      switch activeDetector {
  //    case .onDeviceBarcode:
  //      scanBarcodesOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
      case .onDeviceFace:
        detectFacesOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
  //    case .onDeviceText, .onDeviceTextChinese, .onDeviceTextDevanagari, .onDeviceTextJapanese,
  //      .onDeviceTextKorean:
  //      recognizeTextOnDevice(
  //        in: visionImage, width: imageWidth, height: imageHeight, detectorType: activeDetector)
  ////    case .onDeviceImageLabel:
  //      detectLabels(
  //        in: visionImage, width: imageWidth, height: imageHeight, shouldUseCustomModel: false)
  //    case .onDeviceImageLabelsCustom:
  //      detectLabels(
  //        in: visionImage, width: imageWidth, height: imageHeight, shouldUseCustomModel: true)
      case .onDeviceObjectProminentWithClassifier
        :
        // The `options.detectorMode` defaults to `.stream`
        let options = ObjectDetectorOptions()
        options.shouldEnableClassification = shouldEnableClassification
        options.shouldEnableMultipleObjects = shouldEnableMultipleObjects
        detectObjectsOnDevice(
          in: visionImage,
          width: imageWidth,
          height: imageHeight,
          options: options);
  //    case .onDeviceObjectCustomProminentNoClassifier, .onDeviceObjectCustomProminentWithClassifier,
  //      .onDeviceObjectCustomMultipleNoClassifier, .onDeviceObjectCustomMultipleWithClassifier:
  //      guard
  //        let localModelFilePath = Bundle.main.path(
  //          forResource: Constant.localModelFile.name,
  //          ofType: Constant.localModelFile.type
  //        )
  //      else {
  //        print("Failed to find custom local model file.")
  //        return
  //      }
  //      let localModel = LocalModel(path: localModelFilePath)
  //      // The `options.detectorMode` defaults to `.stream`
  //      let options = CustomObjectDetectorOptions(localModel: localModel)
  //      options.shouldEnableClassification = shouldEnableClassification
  //      options.shouldEnableMultipleObjects = shouldEnableMultipleObjects
  //      detectObjectsOnDevice(
  //        in: visionImage,
  //        width: imageWidth,
  //        height: imageHeight,
  //        options: options)

          
      case .pose:
        detectPose(in: inputImage, width: imageWidth, height: imageHeight)
  //    case .segmentationSelfie:
  //      detectSegmentationMask(in: visionImage, sampleBuffer: sampleBuffer)
      default: break

      }
    }
}

// MARK: - Constants

public enum Detector: String {
 // case onDeviceBarcode = "Barcode Scanning"
case onDeviceFace = "Face Detection"
//  case onDeviceText = "Text Recognition"
//  case onDeviceTextChinese = "Text Recognition Chinese"
//  case onDeviceTextDevanagari = "Text Recognition Devanagari"
//  case onDeviceTextJapanese = "Text Recognition Japanese"
//  case onDeviceTextKorean = "Text Recognition Korean"
//  case onDeviceImageLabel = "Image Labeling"
//  case onDeviceImageLabelsCustom = "Image Labeling Custom"
//  case onDeviceObjectProminentNoClassifier = "ODT, single, no labeling"
  case onDeviceObjectProminentWithClassifier = "Detect Person"
//  case onDeviceObjectMultipleNoClassifier = "ODT, multiple, no labeling"
//  case onDeviceObjectMultipleWithClassifier = "ODT, multiple, labeling"
//  case onDeviceObjectCustomProminentNoClassifier = "ODT, custom, single, no labeling"
//  case onDeviceObjectCustomProminentWithClassifier = "ODT, custom, single, labeling"
//  case onDeviceObjectCustomMultipleNoClassifier = "ODT, custom, multiple, no labeling"
//  case onDeviceObjectCustomMultipleWithClassifier = "ODT, custom, multiple, labeling"
  case pose = "Detect Angle"
//  case poseAccurate = "Pose Detection, accurate"
//  case segmentationSelfie = "Selfie Segmentation"
}
extension CameraViewController: MTKViewDelegate {
    // Called whenever view changes orientation or layout is changed
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawRectResized(size: size)
    }
    
    // Called whenever the view needs to render
    func draw(in view: MTKView) {
        if renderer.isCapture {
            print("draw")
        renderer.draw()
        } else {
            if renderer.isLoadingStarted {
                renderer.getFiles()
                createSpinnerView()
                renderer.isLoadingStarted = false
            }
        }
    }
        
}
protocol RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get set }
    var depthStencilPixelFormat: MTLPixelFormat { get set }
    var sampleCount: Int { get set }
}
extension MTKView: RenderDestinationProvider {
    
}
extension CameraViewController {
    
  func initARSession() {
    guard ARWorldTrackingConfiguration.isSupported else {
      print("*** ARConfig: AR World Tracking Not Supported")
      return
    }
    
    let config = ARWorldTrackingConfiguration()
    config.worldAlignment = .gravity
    config.providesAudioData = false
    config.isLightEstimationEnabled = true
    config.environmentTexturing = .automatic
    session.run(config)
  }
  
  func resetARSession() {
    let config = session.configuration as!
      ARWorldTrackingConfiguration
    config.planeDetection = .horizontal
    session.run(config, options: [.resetTracking, .removeExistingAnchors])
  }
  
  func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
    switch camera.trackingState {
    case .notAvailable:
      self.trackingStatus = "Tracking:  Not available!"
    case .normal:
      self.trackingStatus = ""
    case .limited(let reason):
      switch reason {
      case .excessiveMotion:
        self.trackingStatus = "Tracking: Limited due to excessive motion!"
      case .insufficientFeatures:
        self.trackingStatus = "Tracking: Limited due to insufficient features!"
      case .relocalizing:
        self.trackingStatus = "Tracking: Relocalizing..."
      case .initializing:
        self.trackingStatus = "Tracking: Initializing..."
      @unknown default:
        self.trackingStatus = "Tracking: Unknown..."
      }
    }
  }
  

    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user.
        self.trackingStatus = "AR Session Failure: \(error)"
        guard error is ARError else { return }
        let errorWithInfo = error as NSError
        let messages = [
            errorWithInfo.localizedDescription,
            errorWithInfo.localizedFailureReason,
            errorWithInfo.localizedRecoverySuggestion
        ]
        let errorMessage = messages.compactMap({ $0 }).joined(separator: "\n")
        DispatchQueue.main.async {
            // Present an alert informing about the error that has occurred.
            let alertController = UIAlertController(title: "The AR session failed.", message: errorMessage, preferredStyle: .alert)
            let restartAction = UIAlertAction(title: "Restart Session", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
                if let configuration = self.session.configuration {
                    self.session.run(configuration, options: .resetSceneReconstruction)
                }
            }
            alertController.addAction(restartAction)
            self.present(alertController, animated: true, completion: nil)
        }
    }
    func session(

            _ session: ARSession,

            didUpdate frame: ARFrame

         ) {

             print("get frame")

             

             //var results: [Pose]

             images = session.currentFrame?.capturedImage

             

    //         let image1 = UIImage(ciImage: CIImage(cvPixelBuffer: images!))

             

    //         print("image",images,image1,image1.size.height,image1.size.width)

            // let image = VisionImage(image: image1)

             // Attempt to lock the image buffer to gain access to its memory.

    //         guard CVPixelBufferLockBaseAddress(images!, .readOnly) == kCVReturnSuccess

    //             else {

    //                 return

    //         }

    //         // Create Core Graphics image placeholder.

    //         var image: CGImage?

    //

    //         // Create a Core Graphics bitmap image from the pixel buffer.

    //         VTCreateCGImageFromCVPixelBuffer(images!, options: nil, imageOut: &image)

    //

    //         // Release the image buetectffer.

    //         CVPixelBufferUnlockBaseAddress(images!, .readOnly)

             // Get the CGImage on which to perform requests.

    //         guard let cgImage = image1.cgImage else { return }



             // Create a new image-request handler.
             print("Image",images)
             guard let img_buff = images else{

                 print("no data caught")

                 return

             }

    //         print(img_buff)

             let requestHandler = VNImageRequestHandler(cvPixelBuffer: img_buff)



             // Create a new request to recognize a human body pose.

             let request = VNDetectHumanBodyPoseRequest(completionHandler: bodyPoseHandler)

            print("request", request)



             do {

                 // Perform the body pose-detection request.

                 try requestHandler.perform([request])

             } catch {

                 print("Unable to perform the request: \(error).")

             }

             
             
            // visionImage.orientation = image.imageOrientation

    //         do {

    //             results = try poseDetector!.results(in: image)

    //         } catch let error {

    //           print("Failed to detect pose with error: \(error.localizedDescription).")

    //           return

    //         }

    //         guard let detectedPoses = results, !detectedPoses.isEmpty else {

    //           print("Pose detector returned no results.")

    //           return

    //         }

             

         }
    func imageFromSampleBuffer(sampleBuffer:CMSampleBuffer) -> UIImage? {
            if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                //CVPixelBufferLockBaseAddress(imageBuffer,0)
                let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)
                let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
                let width = CVPixelBufferGetWidth(imageBuffer)
                let height = CVPixelBufferGetHeight(imageBuffer)
                let colorSpace = CGColorSpaceCreateDeviceRGB()
                let context = CGContext(data: baseAddress,width: width,height: height,bitsPerComponent: 8,bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue)

                let quartzImage = context!.makeImage()
                //CVPixelBufferUnlockBaseAddress(imageBuffer,0)

                if let quartzImage = quartzImage {
                    let image = UIImage(cgImage: quartzImage)
                    return image
                }
            }
            return nil
        }
    func bodyPoseHandler(request: VNRequest, error: Error?) {

            guard let observations =

                    request.results as? [VNHumanBodyPoseObservation] else {

                return

            }

            

            // Process each observation to find the recognized body pose points.

            observations.forEach { processObservation($0) }

        }

        func processObservation(_ observation: VNHumanBodyPoseObservation) {

            

            // Retrieve all torso points.

            guard let recognizedPoints =

                    try? observation.recognizedPoints(.torso) else { return }

            

            // Torso joint names in a clockwise ordering.

            let torsoJointNames: [VNHumanBodyPoseObservation.JointName] = [

                .neck,

                .rightShoulder,

                .rightHip,

                .root,

                .leftHip,

                .leftShoulder

            ]

            

            // Retrieve the CGPoints containing the normalized X and Y coordinates.

            let imagePoints: [CGPoint] = torsoJointNames.compactMap {

                guard let point = recognizedPoints[$0], point.confidence > 0 else { return nil }

                

                // Translate the point from normalized-coordinates to image coordinates.

                return VNImagePointForNormalizedPoint(point.location,

                                                      1440,

                                                      1920)

            }
            
            resetManagedLifecycleDetectors(activeDetector: .pose)

            

            // Draw the points onscreen.

           // draw(points: imagePoints)
            //let visionImage = VisionImage(buffer: images as! CMSampleBuffer)
            let orientation = UIUtilities.imageOrientation(
              fromDevicePosition: isUsingFrontCamera ? .front : .back
            )
//            visionImage.orientation = orientation
            print("inside capture")
            //let pixelBuffer: CVImageBuffer? = createPixelBufferFrom(image: images) // see
            let imageWidth = CGFloat(CVPixelBufferGetWidth(images!))
            let imageHeight = CGFloat(CVPixelBufferGetHeight(images!))
            //let sampleBuffer: CMSampleBuffer? = createSampleBufferFrom(pixelBuffer: images!)
           // print("samplebuffer",sampleBuffer)
            
            var sampleBuffer: CMSampleBuffer? = nil

            let scale = CMTimeScale(1_000_000_000)
            let time = CMTime(value: CMTimeValue( 2 * Double(scale)), timescale: scale)
            var timimgInfo: CMSampleTimingInfo = CMSampleTimingInfo( duration: CMTime.invalid,
                                                                     presentationTimeStamp: time,
                                                                     decodeTimeStamp: CMTime.invalid)
            var videoInfo: CMVideoFormatDescription? = nil

//            CMVideoFormatDescriptionCreateForImageBuffer(
//                              allocator: nil,
//                              imageBuffer: images!,
//                              formatDescriptionOut: &videoInfo)
//            print("MyImae1",images)
//            CMSampleBufferCreateForImageBuffer(
//                              allocator: kCFAllocatorDefault,
//                              imageBuffer: images!,
//                              dataReady: true,
//                              makeDataReadyCallback: nil,
//                              refcon: nil,
//                              formatDescription: videoInfo!,
//                              sampleTiming: &timimgInfo,
//                              sampleBufferOut: &sampleBuffer)
//            var image: CGImage?
//            let ciimage = CIImage(cvPixelBuffer: images!)
//            let image1 = self.convert(cmage: ciimage)
//            VTCreateCGImageFromCVPixelBuffer(images!, options: nil, imageOut: &image)
//
//            // Release the image buffer.
//            CVPixelBufferUnlockBaseAddress(images!, .readOnly)
////            currentFrame=image!
    
            let image = CIImage(cvPixelBuffer: images!)
            
                let context = CIContext()
            let cgiImage = context.createCGImage(image, from: image.extent)
            let capturedImage = UIImage(cgImage: cgiImage!)
            
            countframeimage += 1
                if let data = capturedImage.pngData() {
                    let filename = getDocumentsDirectory().appendingPathComponent("copy\(countframeimage).png")
                    try? data.write(to: filename)
                }
            
            print("Image is",capturedImage,capturedImage.description)
           
            guard let inputImage = MLImage(image:capturedImage) else {
              print("Failed to create MLImage from UIImage.")
              return
            }
            
//            guard let inputImage = MLImage(sampleBuffer: sampleBuffer!) else {
//              print("Failed to create MLImage from sample buffer.")
//              return
//            }
////            if inputImage == nil
//            {
//                print("inputImage is Null")
//            }
            print("MyImage2",inputImage)
            DispatchQueue.global(qos: .default).async {
                self.detectPose(in: inputImage,width: imageWidth, height: imageHeight)
            }


        }
    func convert(cmage: CIImage) -> UIImage {
         let context = CIContext(options: nil)
         let cgImage = context.createCGImage(cmage, from: cmage.extent)!
         let image = UIImage(cgImage: cgImage)
         return image
    }
    func createSampleBufferFrom(pixelBuffer: CVPixelBuffer) -> CMSampleBuffer? {
        var sampleBuffer: CMSampleBuffer?
        
        var timimgInfo  = CMSampleTimingInfo()
        var formatDescription: CMFormatDescription? = nil
        CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer, formatDescriptionOut: &formatDescription)
        
        let osStatus = CMSampleBufferCreateReadyWithImageBuffer(
          allocator: kCFAllocatorDefault,
          imageBuffer: pixelBuffer,
          formatDescription: formatDescription!,
          sampleTiming: &timimgInfo,
          sampleBufferOut: &sampleBuffer
        )
        
        // Print out errors
        if osStatus == kCMSampleBufferError_AllocationFailed {
          print("osStatus == kCMSampleBufferError_AllocationFailed")
        }
        if osStatus == kCMSampleBufferError_RequiredParameterMissing {
          print("osStatus == kCMSampleBufferError_RequiredParameterMissing")
        }
        if osStatus == kCMSampleBufferError_AlreadyHasDataBuffer {
          print("osStatus == kCMSampleBufferError_AlreadyHasDataBuffer")
        }
        if osStatus == kCMSampleBufferError_BufferNotReady {
          print("osStatus == kCMSampleBufferError_BufferNotReady")
        }
        if osStatus == kCMSampleBufferError_SampleIndexOutOfRange {
          print("osStatus == kCMSampleBufferError_SampleIndexOutOfRange")
        }
        if osStatus == kCMSampleBufferError_BufferHasNoSampleSizes {
          print("osStatus == kCMSampleBufferError_BufferHasNoSampleSizes")
        }
        if osStatus == kCMSampleBufferError_BufferHasNoSampleTimingInfo {
          print("osStatus == kCMSampleBufferError_BufferHasNoSampleTimingInfo")
        }
        if osStatus == kCMSampleBufferError_ArrayTooSmall {
          print("osStatus == kCMSampleBufferError_ArrayTooSmall")
        }
        if osStatus == kCMSampleBufferError_InvalidEntryCount {
          print("osStatus == kCMSampleBufferError_InvalidEntryCount")
        }
        if osStatus == kCMSampleBufferError_CannotSubdivide {
          print("osStatus == kCMSampleBufferError_CannotSubdivide")
        }
        if osStatus == kCMSampleBufferError_SampleTimingInfoInvalid {
          print("osStatus == kCMSampleBufferError_SampleTimingInfoInvalid")
        }
        if osStatus == kCMSampleBufferError_InvalidMediaTypeForOperation {
          print("osStatus == kCMSampleBufferError_InvalidMediaTypeForOperation")
        }
        if osStatus == kCMSampleBufferError_InvalidSampleData {
          print("osStatus == kCMSampleBufferError_InvalidSampleData")
        }
        if osStatus == kCMSampleBufferError_InvalidMediaFormat {
          print("osStatus == kCMSampleBufferError_InvalidMediaFormat")
        }
        if osStatus == kCMSampleBufferError_Invalidated {
          print("osStatus == kCMSampleBufferError_Invalidated")
        }
        if osStatus == kCMSampleBufferError_DataFailed {
          print("osStatus == kCMSampleBufferError_DataFailed")
        }
        if osStatus == kCMSampleBufferError_DataCanceled {
          print("osStatus == kCMSampleBufferError_DataCanceled")
        }
        
        guard let buffer = sampleBuffer else {
          print("Cannot create sample buffer")
          return nil
        }
        
        return buffer
      }
    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
  func sessionWasInterrupted(_ session: ARSession) {
    self.trackingStatus = "AR Session Was Interrupted!"
  }
  
  func sessionInterruptionEnded(_ session: ARSession) {
    self.trackingStatus = "AR Session Interruption Ended"
  }
}

private enum Constant {
  static let alertControllerTitle = "Vision Detectors"
  static let alertControllerMessage = "Select a detector"
  static let cancelActionTitleText = "Cancel"
  static let videoDataOutputQueueLabel = "com.google.mlkit.visiondetector.VideoDataOutputQueue"
  static let sessionQueueLabel = "com.google.mlkit.visiondetector.SessionQueue"
  static let noResultsMessage = "No Results"
  static let localModelFile = (name: "bird", type: "tflite")
  static let labelConfidenceThreshold = 0.75
  static let smallDotRadius: CGFloat = 4.0
  static let lineWidth: CGFloat = 3.0
  static let originalScale: CGFloat = 1.0
  static let padding: CGFloat = 10.0
  static let resultsLabelHeight: CGFloat = 200.0
  static let resultsLabelLines = 5
  static let imageLabelResultFrameX = 0.4
  static let imageLabelResultFrameY = 0.1
  static let imageLabelResultFrameWidth = 0.5
  static let imageLabelResultFrameHeight = 0.8
  static let segmentationMaskAlpha: CGFloat = 0.5
}
