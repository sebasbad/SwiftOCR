//
//  SwiftOCR.swift
//  SwiftOCR
//
//  Created by Nicolas Camenisch on 18.04.16.
//  Copyright © 2016 Nicolas Camenisch. All rights reserved.
//

#if os(iOS)
    import UIKit
#else
    import Cocoa
#endif

import CoreGraphics

import GPUImage

internal let globalNetwork = FFNN.fromFile(NSBundle(forClass: SwiftOCR.self).URLForResource("OCR-Network", withExtension: nil, subdirectory: nil, localization: nil)!) ?? FFNN(inputs: 321, hidden: 100, outputs: 36, learningRate: 0.7, momentum: 0.4, weights: nil, activationFunction: .Sigmoid, errorFunction: .CrossEntropy(average: false))

public class SwiftOCR {
    
    #if os(iOS)
    ///The image used for OCR
    public var image:UIImage?
    #else
    ///The image used for OCR
    public var image:NSImage?
    #endif
    
    private  let network = globalNetwork.copy()
    
    public   var delegate:SwiftOCRDelegate?
    public   var currentOCRRecognizedBlobs = [SwiftOCRRecognizedBlob]()
    
    public   init(){}
    
    /**
     
     Performs ocr on `SwiftOCR().image`.
     
     - Parameter completionHandler: The completion handler that gets invoked after the ocr is finished.
     
     */
    
    public   func recognize(completionHandler: (String) -> Void){
        
        let confidenceThreshold:Float = 0.1 //Confidence must be bigger than the threshold
        
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), {
            if let imageToRecognize = self.image {
                #if os(iOS)
                    var preprocessedImage: UIImage!
                #else
                    var preprocessedImage: NSImage!
                #endif
                
                if let preprocessFunc = self.delegate?.preprocessImageForOCR, let processedImage = preprocessFunc(imageToRecognize){
                    preprocessedImage = processedImage
                } else {
                    preprocessedImage = self.preprocessImageForOCR(self.image)
                }
                
                let blobs                  = self.extractBlobs(preprocessedImage)
                var recognizedString       = ""
                var ocrRecognizedBlobArray = [SwiftOCRRecognizedBlob]()
                
                for blob in blobs {
                    
                    do {
                        let blobData       = self.convertImageToFloatArray(blob.0, resize: true)
                        let networkResult  = try self.network.update(inputs: blobData)
                        
                        if networkResult.maxElement() >= confidenceThreshold {
                            let recognizedChar = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".characters)[networkResult.indexOf(networkResult.maxElement() ?? 0) ?? 0]
                            recognizedString.append(recognizedChar)
                        }
                        
                        //Generate SwiftOCRRecognizedBlob
                        
                        var ocrRecognizedBlobCharactersWithConfidenceArray = [(character: Character, confidence: Float)]()
                        let ocrRecognizedBlobConfidenceThreshold = networkResult.reduce(0, combine: +)/Float(networkResult.count)
                        
                        for networkResultIndex in 0..<networkResult.count {
                            let characterConfidence = networkResult[networkResultIndex]
                            let character           = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".characters)[networkResultIndex]
                            
                            if characterConfidence >= ocrRecognizedBlobConfidenceThreshold {
                                ocrRecognizedBlobCharactersWithConfidenceArray.append((character: character, confidence: characterConfidence))
                            }
                            
                        }
                        
                        let currentRecognizedBlob = SwiftOCRRecognizedBlob(charactersWithConfidence: ocrRecognizedBlobCharactersWithConfidenceArray, boundingBox: blob.1)
                        
                        ocrRecognizedBlobArray.append(currentRecognizedBlob)
                        
                    } catch {
                        print(error)
                    }
                    
                }
                
                self.currentOCRRecognizedBlobs = ocrRecognizedBlobArray
                completionHandler(recognizedString)
                
            } else {
                print("You first have to set a SwiftOCR().image")
                completionHandler(String())
            }
        })
        
    }
    
    #if os(iOS)
    
    /**
     
     Extracts the characters using [Connected-component labeling](https://en.wikipedia.org/wiki/Connected-component_labeling).
     
     - Parameter image: The image which will be used for the connected-component labeling. If you pass in nil, the `SwiftOCR().image` will be used.
     - Returns:         An array containing the etracted and cropped Blobs and their bounding box.
     
     */
    
    internal func extractBlobs(image:UIImage?) -> [(UIImage, CGRect)] {
        if let inputImage = image {
            let pixelData = CGDataProviderCopyData(CGImageGetDataProvider(inputImage.CGImage))
            let bitmapData: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
            
            //data <- bitmapData
            
            let numberOfComponents = CGImageGetBitsPerPixel(inputImage.CGImage) / CGImageGetBitsPerComponent(inputImage.CGImage)
            
            let inputImageHeight   = inputImage.size.height
            let inputImageWidth    = inputImage.size.width
            
            var data = [UInt16](count: (Int(inputImageWidth) * Int(inputImageHeight)) * numberOfComponents, repeatedValue: 0)
            
            for dataIndex in 0..<data.count {
                data[dataIndex] = UInt16(round(Double(bitmapData[dataIndex])/255)*255)
            }
            
            //First Pass
            
            var currentLabel:UInt16 = 0 {
                didSet {
                    if currentLabel == 255 {
                        currentLabel = 256
                    }
                }
            }
            var labelsUnion = UnionFind<UInt16>()
            
            for y in 0..<Int(inputImageHeight) {
                for x in 0..<Int(inputImageWidth) {
                    let pixelInfo: Int = ((Int(inputImageWidth) * Int(y)) + Int(x)) * numberOfComponents
                    let pixelIndex:(Int, Int) -> Int = {x, y in
                        return ((Int(inputImageWidth) * Int(y)) + Int(x)) * numberOfComponents
                    }
                    
                    if data[pixelInfo] == 0 { //Is Black
                        if x == 0 { //Left no pixel
                            if y == 0 { //Top no pixel
                                currentLabel += 1
                                labelsUnion.addSetWith(currentLabel)
                                data[pixelInfo] = currentLabel
                            } else if y > 0 { //Top pixel
                                if data[pixelIndex(x, y-1)] != 255 { //Top Label
                                    data[pixelInfo] = data[pixelIndex(x, y-1)]
                                } else { //Top no Label
                                    currentLabel += 1
                                    labelsUnion.addSetWith(currentLabel)
                                    data[pixelInfo] = currentLabel
                                }
                            }
                        } else { //Left pixel
                            if y == 0 { //Top no pixel
                                if data[pixelIndex(x-1,y)] != 255 { //Left Label
                                    data[pixelInfo] = data[pixelIndex(x-1,y)]
                                } else { //Left no Label
                                    currentLabel += 1
                                    labelsUnion.addSetWith(currentLabel)
                                    data[pixelInfo] = currentLabel
                                }
                            } else if y > 0 { //Top pixel
                                if data[pixelIndex(x-1,y)] != 255 { //Left Label
                                    if data[pixelIndex(x,y-1)] != 255 { //Top Label
                                        
                                        if data[pixelIndex(x,y-1)] != data[pixelIndex(x-1,y)] {
                                            labelsUnion.unionSetsContaining(data[pixelIndex(x,y-1)], and: data[pixelIndex(x-1,y)])
                                        }
                                        
                                        data[pixelInfo] = data[pixelIndex(x,y-1)]
                                    } else { //Top no Label
                                        data[pixelInfo] = data[pixelIndex(x-1,y)]
                                    }
                                } else { //Left no Label
                                    if data[pixelIndex(x,y-1)] != 255 { //Top Label
                                        data[pixelInfo] = data[pixelIndex(x,y-1)]
                                    } else { //Top no Label
                                        currentLabel += 1
                                        labelsUnion.addSetWith(currentLabel)
                                        data[pixelInfo] = currentLabel
                                    }
                                }
                            }
                        }
                    }
                    
                }
            }
            
            //Second Pass
            
            let parentArray = Array(labelsUnion.parent.uniq())
            
            for y in 0..<Int(inputImageHeight) {
                for x in 0..<Int(inputImageWidth) {
                    let pixelInfo: Int = ((Int(inputImageWidth) * Int(y)) + Int(x)) * numberOfComponents
                    
                    if data[pixelInfo] != 255 {
                        data[pixelInfo] = UInt16(parentArray.indexOf(labelsUnion.setOf(data[pixelInfo]) ?? 0) ?? 0)
                    }
                    
                }
            }
            
            //Extract Images
            
            //Merge rects
            
            var mergeUnion = UnionFind<UInt16>()
            var mergeLabelRects = [CGRect]()
            
            let xMergeRadius:CGFloat = 1
            let yMergeRadius:CGFloat = 3
            
            for label in 0..<parentArray.count {
                var minX = Int(inputImageWidth)
                var maxX = 0
                var minY = Int(inputImageHeight)
                var maxY = 0
                
                for y in 0..<Int(inputImageHeight) {
                    for x in 0..<Int(inputImageWidth) {
                        let pixelInfo: Int = ((Int(inputImageWidth) * Int(y)) + Int(x)) * numberOfComponents
                        
                        if data[pixelInfo] == UInt16(label) {
                            minX = min(minX, x)
                            maxX = max(maxX, x)
                            minY = min(minY, y)
                            maxY = max(maxY, y)
                        }
                        
                    }
                }
                
                //Filter blobs
                
                let minMaxCorrect = (minX < maxX && minY < maxY)
                let correctFormat:Bool = {
                    if (maxY - minY) != 0 {
                        return Double(maxX - minX)/Double(maxY - minY) < 1.6
                    } else {
                        return false
                    }
                }()
                
                let notToTall    = Double(maxY - minY) < Double(inputImage.size.height) * 0.75
                let notToWide    = Double(maxX - minX) < Double(inputImage.size.width ) * 0.25
                let notToShort   = Double(maxY - minY) > Double(inputImage.size.height) * 0.25
                let notToThin    = Double(maxX - minX) > Double(inputImage.size.width ) * 0.01
                
                let notToSmall   = (maxX - minX)*(maxY - minY) > 100
                
                let positionIsOK = minY != 0 && minX != 0 && maxY != Int(inputImageHeight - 1) && maxX != Int(inputImageWidth - 1)
                
                if minMaxCorrect && correctFormat && notToTall && notToWide && notToShort && notToThin && notToSmall && positionIsOK{
                    let labelRect = CGRectMake(CGFloat(CGFloat(minX) - xMergeRadius), CGFloat(CGFloat(minY) - yMergeRadius), CGFloat(CGFloat(maxX - minX) + 2*xMergeRadius + 1), CGFloat(CGFloat(maxY - minY) + 2*yMergeRadius + 1))
                    mergeUnion.addSetWith(UInt16(label))
                    mergeLabelRects.append(labelRect)
                }
            }
            
            for rectOneIndex in 0..<mergeLabelRects.count {
                for rectTwoIndex in 0..<mergeLabelRects.count {
                    if mergeLabelRects[rectOneIndex].intersects(mergeLabelRects[rectTwoIndex]) && rectOneIndex != rectTwoIndex{
                        mergeUnion.unionSetsContaining(UInt16(rectOneIndex), and: UInt16(rectTwoIndex))
                        mergeLabelRects[rectOneIndex].unionInPlace(mergeLabelRects[rectTwoIndex])
                    }
                }
            }
            
            var outputImages = [(UIImage, CGRect)]()
            
            mergeLabelRects.uniqInPlace()
            
            //Extract images
            
            for rect in mergeLabelRects {
                let cropRect = rect.insetBy(dx: CGFloat(xMergeRadius), dy: CGFloat(yMergeRadius))
                if let croppedCGImage = CGImageCreateWithImageInRect(inputImage.CGImage, cropRect) {
                    let croppedImage = UIImage(CGImage: croppedCGImage)
                    outputImages.append((croppedImage, cropRect))
                }
            }
            
            outputImages.sortInPlace({return $0.0.1.origin.x < $0.1.1.origin.x})
            return outputImages
        } else {
            return []
        }
    }
    
    /**
     
     Takes an array of images and then resized them to **16x20px**. This is the standard size for the input for the neural network.
     
     - Parameter blobImages: The array of images that should get resized.
     - Returns:              An array containing the resized images.
     
     */
    
    internal func resizeBlobs(blobImages: [UIImage]) -> [UIImage] {
        
        var resizedBlobs = [UIImage]()
        
        for blobImage in blobImages {
            let cropSize = CGSizeMake(16, 20)
            
            //Downscale
            let cgImage = blobImage.CGImage
            
            let width = cropSize.width
            let height = cropSize.height
            let bitsPerComponent = 8
            let bytesPerRow = 0
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGImageAlphaInfo.NoneSkipLast.rawValue
            
            let context = CGBitmapContextCreate(nil, Int(width), Int(height), bitsPerComponent, bytesPerRow, colorSpace, bitmapInfo)
            
            CGContextSetInterpolationQuality(context, CGInterpolationQuality.None)
            
            CGContextDrawImage(context, CGRectMake(0, 0, cropSize.width, cropSize.height), cgImage)
            
            let resizedCGImage = CGImageCreateWithImageInRect(CGBitmapContextCreateImage(context), CGRectMake(0, 0, cropSize.width, cropSize.height))!
            
            let resizedNSImage = UIImage(CGImage: resizedCGImage)
            resizedBlobs.append(resizedNSImage)
        }
        
        return resizedBlobs
        
    }
    
    /**
     
     Uses the default preprocessing algorithm to binarize the image. It uses the [GPUImage framework](https://github.com/BradLarson/GPUImage).
     
     - Parameter inputImage: The image which should be binarized. If you pass in nil, the `SwiftOCR().image` will be used.
     - Retuns:               The binarized image.
     
     */
    
    public func preprocessImageForOCR(inputImage:UIImage?) -> UIImage? {
        
        if let image = inputImage ?? image {
            
            let grayScaleFilter     = Luminance()
            let grayScaleFilterTwo  = Luminance()
            let invertFilter        = ColorInversion()
            let blurFilter          = SingleComponentGaussianBlur()
            let dodgeFilter         = ColorDodgeBlend()
            
            blurFilter.blurRadiusInPixels  = 10
            
            let medianFilter           = MedianFilter()
            let openingFilter          = OpeningFilter()
            let bilateralFilter        = BilateralBlur()
            let firstBrightnessFilter  = BrightnessAdjustment()
            let contrastFilter         = ContrastAdjustment()
            let secondBrightnessFilter = BrightnessAdjustment()
            
            bilateralFilter.distanceNormalizationFactor = 2
            firstBrightnessFilter.brightness            = -0.28
            contrastFilter.contrast                     = 2.35
            secondBrightnessFilter.brightness           = -0.08
            
            return image.filterWithPipeline{input, output in
                input --> grayScaleFilter --> invertFilter --> dodgeFilter
                input --> grayScaleFilterTwo --> blurFilter --> dodgeFilter --> medianFilter --> openingFilter --> bilateralFilter --> firstBrightnessFilter --> contrastFilter --> secondBrightnessFilter --> output
            }
            
        } else {
            return nil
        }
        
    }
    
    /**
     
     Takes an image and converts it to an array of floats. The array gets generated by taking the pixel-data of the red channel and then converting it into floats. This array can be used as input for the neural network.
     
     - Parameter image:  The image which should get converted to the float array.
     - Parameter resize: If you set this to true, the image firste gets resized. The default value is `true`.
     - Retuns:           The array containing the pixel-data of the red channel.
     
     */
    
    internal func convertImageToFloatArray(image: UIImage, resize: Bool = true) -> [Float] {
        
        let resizedBlob: UIImage = {
            if resize {
                return resizeBlobs([image]).first!
            } else {
                return image
            }
        }()
        
        let pixelData = CGDataProviderCopyData(CGImageGetDataProvider(resizedBlob.CGImage))
        let bitmapData: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        
        var imageData = [Float]()
        
        for y in 0..<Int(resizedBlob.size.height) {
            for x in 0..<Int(resizedBlob.size.width) {
                let pixelInfo: Int = ((Int(resizedBlob.size.width) * Int(y)) + Int(x)) * 4
                imageData.append(Float(bitmapData[pixelInfo])/255)
            }
        }
        
        let aspectRactio = Float(image.size.width / image.size.height)
        
        imageData.append(aspectRactio)
        
        return imageData
    }
    
    #else
    
    /**
     
     Extracts the characters using [Connected-component labeling](https://en.wikipedia.org/wiki/Connected-component_labeling).
     
     - Parameter image: The image which will be used for the connected-component labeling. If you pass in nil, the SwiftOCR().image will be used.
     - Returns:         An array containing the etracted and cropped Blobs and their bounding box.
     
     */
    
    internal func extractBlobs(image:NSImage?) -> [(NSImage, CGRect)] {
        
        if let inputImage = image ?? self.image {
            let bitmapRep = NSBitmapImageRep(data: inputImage.TIFFRepresentation!)!
            let bitmapData: UnsafeMutablePointer<UInt8> = bitmapRep.bitmapData
            
            let cgImage = bitmapRep.CGImage
            //data <- bitmapData
            let inputImageHeight = inputImage.size.height
            let inputImageWidth  = inputImage.size.width
            
            var data = [UInt16](count: (Int(inputImageWidth) * Int(inputImageHeight)) * bitmapRep.samplesPerPixel, repeatedValue: 0)
            
            for y in 0..<Int(inputImageHeight) {
                for x in 0..<Int(inputImageWidth) {
                    let pixelInfo: Int = ((Int(inputImageWidth) * Int(y)) + Int(x)) * bitmapRep.samplesPerPixel
                    for i in 0..<bitmapRep.samplesPerPixel {
                        data[pixelInfo+i] = UInt16(bitmapData[pixelInfo+i])
                    }
                }
            }
            
            //First Pass
            
            var currentLabel:UInt16 = 0 {
                didSet {
                    if currentLabel == 255 {
                        currentLabel = 256
                    }
                }
            }
            var labelsUnion = UnionFind<UInt16>()
            
            for y in 0..<Int(inputImageHeight) {
                for x in 0..<Int(inputImageWidth) {
                    let pixelInfo: Int = ((Int(inputImageWidth) * Int(y)) + Int(x)) * bitmapRep.samplesPerPixel
                    let pixelIndex:(Int, Int) -> Int = {x, y in
                        return ((Int(inputImageWidth) * Int(y)) + Int(x))  * bitmapRep.samplesPerPixel
                    }
                    
                    if data[pixelInfo] == 0 { //Is Black
                        if x == 0 { //Left no pixel
                            if y == 0 { //Top no pixel
                                currentLabel += 1
                                labelsUnion.addSetWith(currentLabel)
                                data[pixelInfo] = currentLabel
                            } else if y > 0 { //Top pixel
                                if data[pixelIndex(x, y-1)] != 255 { //Top Label
                                    data[pixelInfo] = data[pixelIndex(x, y-1)]
                                } else { //Top no Label
                                    currentLabel += 1
                                    labelsUnion.addSetWith(currentLabel)
                                    data[pixelInfo] = currentLabel
                                }
                            }
                        } else { //Left pixel
                            if y == 0 { //Top no pixel
                                if data[pixelIndex(x-1,y)] != 255 { //Left Label
                                    data[pixelInfo] = data[pixelIndex(x-1,y)]
                                } else { //Left no Label
                                    currentLabel += 1
                                    labelsUnion.addSetWith(currentLabel)
                                    data[pixelInfo] = currentLabel
                                }
                            } else if y > 0 { //Top pixel
                                if data[pixelIndex(x-1,y)] != 255 { //Left Label
                                    if data[pixelIndex(x,y-1)] != 255 { //Top Label
                                        
                                        if data[pixelIndex(x,y-1)] != data[pixelIndex(x-1,y)] {
                                            labelsUnion.unionSetsContaining(data[pixelIndex(x,y-1)], and: data[pixelIndex(x-1,y)])
                                        }
                                        
                                        data[pixelInfo] = data[pixelIndex(x,y-1)]
                                    } else { //Top no Label
                                        data[pixelInfo] = data[pixelIndex(x-1,y)]
                                    }
                                } else { //Left no Label
                                    if data[pixelIndex(x,y-1)] != 255 { //Top Label
                                        data[pixelInfo] = data[pixelIndex(x,y-1)]
                                    } else { //Top no Label
                                        currentLabel += 1
                                        labelsUnion.addSetWith(currentLabel)
                                        data[pixelInfo] = currentLabel
                                    }
                                }
                            }
                        }
                    }
                    
                }
            }
            
            //Second Pass
            
            let parentArray = Array(labelsUnion.parent.uniq())
            
            for y in 0..<Int(inputImageHeight) {
                for x in 0..<Int(inputImageWidth) {
                    let pixelInfo: Int = ((Int(inputImageWidth) * Int(y)) + Int(x)) * bitmapRep.samplesPerPixel
                    
                    if data[pixelInfo] != 255 {
                        data[pixelInfo] = UInt16(parentArray.indexOf(labelsUnion.setOf(data[pixelInfo]) ?? 0) ?? 0) // * UInt16(255/parentArray.count)
                    }
                    
                }
            }
            
            //Extract Images
            
            //Merge rects
            
            var mergeUnion = UnionFind<UInt16>()
            var mergeLabelRects = [CGRect]()
            
            let xMergeRadius:CGFloat = 1
            let yMergeRadius:CGFloat = 3
            
            for label in 0..<parentArray.count {
                var minX = Int(inputImageWidth)
                var maxX = 0
                var minY = Int(inputImageHeight)
                var maxY = 0
                
                for y in 0..<Int(inputImageHeight) {
                    for x in 0..<Int(inputImageWidth) {
                        let pixelInfo: Int = ((Int(inputImageWidth) * Int(y)) + Int(x)) * bitmapRep.samplesPerPixel
                        
                        if data[pixelInfo] == UInt16(label) {
                            minX = min(minX, x)
                            maxX = max(maxX, x)
                            minY = min(minY, y)
                            maxY = max(maxY, y)
                        }
                        
                    }
                }
                
                //Filter blobs
                
                let minMaxCorrect = (minX < maxX && minY < maxY)
                let correctFormat:Bool = {
                    if (maxY - minY) != 0 {
                        return Double(maxX - minX)/Double(maxY - minY) < 1.6
                    } else {
                        return false
                    }
                }()
                
                let notToTall    = Double(maxY - minY) < Double(inputImage.size.height) * 0.75
                let notToWide    = Double(maxX - minX) < Double(inputImage.size.width ) * 0.25
                let notToShort   = Double(maxY - minY) > Double(inputImage.size.height) * 0.25
                let notToThin    = Double(maxX - minX) > Double(inputImage.size.width ) * 0.01
                
                let notToSmall   = (maxX - minX)*(maxY - minY) > 100
                
                let positionIsOK = minY != 0 && minX != 0 && maxY != Int(inputImageHeight - 1) && maxX != Int(inputImageWidth - 1)
                
                if minMaxCorrect && correctFormat && notToTall && notToWide && notToShort && notToThin && notToSmall && positionIsOK{
                    let labelRect = CGRectMake(CGFloat(CGFloat(minX) - xMergeRadius), CGFloat(CGFloat(minY) - yMergeRadius), CGFloat(CGFloat(maxX - minX) + 2*xMergeRadius + 1), CGFloat(CGFloat(maxY - minY) + 2*yMergeRadius + 1))
                    mergeUnion.addSetWith(UInt16(label))
                    mergeLabelRects.append(labelRect)
                }
            }
            
            for rectOneIndex in 0..<mergeLabelRects.count {
                for rectTwoIndex in 0..<mergeLabelRects.count {
                    if mergeLabelRects[rectOneIndex].intersects(mergeLabelRects[rectTwoIndex]) && rectOneIndex != rectTwoIndex{
                        mergeUnion.unionSetsContaining(UInt16(rectOneIndex), and: UInt16(rectTwoIndex))
                        mergeLabelRects[rectOneIndex].unionInPlace(mergeLabelRects[rectTwoIndex])
                    }
                }
            }
            
            var outputImages = [(NSImage, CGRect)]()
            
            mergeLabelRects.uniqInPlace()
            
            //Extract images
            
            for rect in mergeLabelRects {
                let cropRect = rect.insetBy(dx: CGFloat(xMergeRadius), dy: CGFloat(yMergeRadius))
                if let croppedCGImage = CGImageCreateWithImageInRect(cgImage, cropRect) {
                    let croppedImage = NSImage(CGImage: croppedCGImage, size: cropRect.size)
                    print(croppedImage)
                    outputImages.append((croppedImage, cropRect))
                }
            }
            
            outputImages.sortInPlace({return $0.0.1.origin.x < $0.1.1.origin.x})
            return outputImages
        } else {
            return []
        }
    }
    
    /**
     
     Takes an array of images and then resized them to **16x20px**. This is the standard size for the input for the neural network.
     
     - Parameter blobImages: The array of images that should get resized.
     - Returns:              An array containing the resized images.
     
     */
    
    internal func resizeBlobs(blobImages: [NSImage]) -> [NSImage] {
        
        var resizedBlobs = [NSImage]()
        
        for blobImage in blobImages {
            let cropSize = CGSizeMake(16, 20)
            
            //Downscale
            let cgImage = NSBitmapImageRep(data: blobImage.TIFFRepresentation!)!.CGImage!
            
            let width = cropSize.width
            let height = cropSize.height
            let bitsPerComponent = 8
            let bytesPerRow = 0
            let colorSpace = CGColorSpaceCreateDeviceGray()
            let bitmapInfo = kColorSyncAlphaNone.rawValue
            
            let context = CGBitmapContextCreate(nil, Int(width), Int(height), bitsPerComponent, bytesPerRow, colorSpace, bitmapInfo)
            
            CGContextSetInterpolationQuality(context, CGInterpolationQuality.None)
            
            CGContextDrawImage(context, CGRectMake(0, 0, cropSize.width, cropSize.height), cgImage)
            
            let resizedCGImage = CGImageCreateWithImageInRect(CGBitmapContextCreateImage(context), CGRectMake(0, 0, cropSize.width, cropSize.height))!
            
            let resizedNSImage = NSImage(CGImage: resizedCGImage, size: NSSize(width: cropSize.width, height:cropSize.height))
            resizedBlobs.append(resizedNSImage)
        }
        
        return resizedBlobs
        
    }
    
    /**
     
     Uses the default preprocessing algorithm to binarize the image. It uses the [GPUImage framework](https://github.com/BradLarson/GPUImage).
     
     - Parameter inputImage: The image which should be binarized. If you pass in nil, the SwiftOCR().image will be used.
     - Retuns:               The binarized image.
     
     */
    
    public func preprocessImageForOCR(inputImage:NSImage?) -> NSImage? {
        
        if let image = inputImage ?? image {
            
            let saturationFilter    = SaturationAdjustment()
            let saturationFilterTwo = SaturationAdjustment()
            let invertFilter        = ColorInversion()
            let blurFilter          = BoxBlur()
            let dodgeFilter         = ColorDodgeBlend()
            
            saturationFilter.saturation    = -2.0
            saturationFilterTwo.saturation = -2.0
            blurFilter.blurRadiusInPixels  = 10
            
            let medianFilter           = MedianFilter()
            let openingFilter          = OpeningFilter()
            let bilateralFilter        = BilateralBlur()
            let firstBrightnessFilter  = BrightnessAdjustment()
            let contrastFilter         = ContrastAdjustment()
            let secondBrightnessFilter = BrightnessAdjustment()
            let thresholdFilter        = LuminanceThreshold()
            
            bilateralFilter.distanceNormalizationFactor = 1.6
            firstBrightnessFilter.brightness            = -0.28
            contrastFilter.contrast                     = 2.35
            secondBrightnessFilter.brightness           = -0.08
            thresholdFilter.threshold                   = 0.5
            
            
            let processedImage = image.filterWithPipeline{input, output in
                input --> saturationFilter --> dodgeFilter
                input --> saturationFilterTwo --> invertFilter --> blurFilter --> dodgeFilter --> medianFilter --> openingFilter --> bilateralFilter --> firstBrightnessFilter --> contrastFilter --> secondBrightnessFilter --> thresholdFilter --> output
            }
            
            
            return processedImage
        } else {
            return nil
        }
        
    }
    
    
    /**
     
     Takes an image and converts it to an array of floats. The array gets generated by taking the pixel-data of the red channel and then converting it into floats. This array can be used as input for the neural network.
     
     - Parameter image:  The image which should get converted to the float array.
     - Parameter resize: If you set this to true, the image firste gets resized. The default value is `true`.
     - Retuns:           The array containing the pixel-data of the red channel.
     
     */
    
    internal func convertImageToFloatArray(image: NSImage, resize: Bool = true) -> [Float] {
        
        let resizedBlob: NSImage = {
            if resize {
                return resizeBlobs([image]).first!
            } else {
                return image
            }
        }()
        
        let bitmapRep  = NSBitmapImageRep(data: resizedBlob.TIFFRepresentation!)!
        let bitmapData = bitmapRep.bitmapData
        
        var imageData = [Float]()
        
        for y in 0..<Int(resizedBlob.size.height) {
            for x in 0..<Int(resizedBlob.size.width) {
                let pixelInfo: Int = ((Int(resizedBlob.size.width) * Int(y)) + Int(x)) * bitmapRep.samplesPerPixel
                imageData.append(Float(bitmapData[pixelInfo])/255)
            }
        }
        
        let aspectRactio = Float(image.size.width / image.size.height)
        
        imageData.append(aspectRactio)
        
        return imageData
    }
    
    #endif
    
}

@objc public protocol SwiftOCRDelegate {
    
    #if os(iOS)
    
    /**
     
     Implement this methode for a custom image preprocessing algorithm. Only return a binary image.
     
     - Parameter inputImage: The image to preprocess.
     - Returns:              The preprocessed, binarized image that SwiftOCR should use for OCR. If you return nil SwiftOCR will use its default preprocessing algorithm.
     
     */
    
    optional func preprocessImageForOCR(inputImage: UIImage) -> UIImage?
    
    #else
    
    /**
     
     Implement this methode for a custom image preprocessing algorithm. Only return a binary image.
     
     - Parameter inputImage: The image to preprocess.
     - Returns:              The preprocessed, binarized image that SwiftOCR should use for OCR. If you return nil SwiftOCR will use its standard preprocessing algorithm.
     
     */
    
    
    optional func preprocessImageForOCR(inputImage: NSImage) -> NSImage?
    
    #endif
}

public struct SwiftOCRRecognizedBlob {
    
    let charactersWithConfidence: [(character: Character, confidence: Float)]!
    let boundingBox:              CGRect!
    
    init(charactersWithConfidence: [(character: Character, confidence: Float)]!, boundingBox: CGRect) {
        self.charactersWithConfidence = charactersWithConfidence.sort({return $0.0.confidence > $0.1.confidence})
        self.boundingBox = boundingBox
    }
    
}
