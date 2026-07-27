import qupath.ext.instanseg.core.InstanSeg
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.common.GeneralTools

// ======================================
// LOAD IMAGE
// ======================================
def imageData = getCurrentImageData()
if (imageData == null) {
    println "ERROR: No image loaded"
    return
}
def server = imageData.getServer()

println "======================================"
println "STARTING ANALYSIS"
println "======================================"
println "Image: " + server.getMetadata().getName()
println "Width: " + server.getWidth()
println "Height: " + server.getHeight()

// ======================================
// CLEAR OBJECTS
// ======================================
removeAllObjects()

// ======================================
// SET IMAGE TYPE
// ======================================
setImageType('BRIGHTFIELD_H_DAB')

// ======================================
// AUTO MAGNIFICATION FROM FILENAME
// ======================================
def imageNameLower = server.getMetadata().getName().toLowerCase()
double pixelSize

if (imageNameLower.contains("40x")) {
    pixelSize = 0.25
    println "Magnification: 40x → pixel size 0.25 um/px"
} else if (imageNameLower.contains("20x")) {
    pixelSize = 0.50
    println "Magnification: 20x → pixel size 0.50 um/px"
} else {
    pixelSize = 0.50
    println "Magnification: 10x (default) → pixel size 0.50 um/px"
}

setPixelSizeMicrons(pixelSize, pixelSize)

// ======================================
// CREATE FULL IMAGE ANNOTATION
// ======================================
def roi = ROIs.createRectangleROI(
    0, 0,
    server.getWidth(),
    server.getHeight(),
    ImagePlane.getDefaultPlane()
)
def annotation = PathObjects.createAnnotationObject(roi)
addObject(annotation)
selectObjects(annotation)
println "Full image annotation created"

// ======================================
// RUN INSTANSEG
// ======================================
println "Running InstanSeg..."

def modelPath = System.getenv("INSTANSEG_MODEL_PATH")
if (modelPath == null || modelPath.trim().isEmpty()) {
    modelPath = new File(System.getProperty("user.home"), "QuPath/v0.7/instanseg/downloaded/brightfield_nuclei-0.1.1").getAbsolutePath()
}

def instanseg = InstanSeg.builder()
    .modelPath(modelPath)
    .device("mps")
    .nThreads(4)
    .tileDims(512)
    .interTilePadding(32)
    .makeMeasurements(true)
    .randomColors(false)
    .build()

instanseg.detectObjects()
println "InstanSeg completed"

// ======================================
// FETCH DETECTIONS
// ======================================
def detections = getDetectionObjects()
println "Total cells detected: " + detections.size()

if (detections.isEmpty()) {
    println "WARNING: No detections found"
    return
}

// ======================================
// DAB CLASSIFICATION
// Fixed OD threshold of 0.2 — validated against QuPath GUI
// Matches published IHC literature standard
// ======================================
double threshold = 0.2
println "DAB threshold: " + threshold + " (fixed OD)"

def positiveClass = getPathClass("Positive")
def negativeClass = getPathClass("Negative")

int positiveCount = 0
int negativeCount = 0

detections.each { cell ->
    def dab = cell.getMeasurementList().get("DAB: Mean")
    if (dab != null && !dab.isNaN() && dab > threshold) {
        cell.setPathClass(positiveClass)
        positiveCount++
    } else {
        cell.setPathClass(negativeClass)
        negativeCount++
    }
}

fireHierarchyUpdate()

double positivityPct = (positiveCount * 100.0) / detections.size()

println "======================================"
println "FINAL RESULTS"
println "======================================"
println "Total cells:    " + detections.size()
println "Positive cells: " + positiveCount
println "Negative cells: " + negativeCount
println "Positivity %:   " + String.format("%.2f", positivityPct)

// ======================================
// OUTPUT DIRECTORY
// ======================================
def outDir = new File(System.getProperty("user.home"), "Desktop/ihc_results")
if (!outDir.exists()) outDir.mkdirs()
def imageName = GeneralTools.stripExtension(server.getMetadata().getName())

// ======================================
// EXPORT CSV
// ======================================
def csvPath = new File(outDir, imageName + "_detections.csv").getAbsolutePath()
saveDetectionMeasurements(csvPath)
println "CSV exported to: " + csvPath

// ======================================
// EXPORT JSON SUMMARY
// ======================================
def jsonPath = new File(outDir, imageName + "_summary.json").getAbsolutePath()
def summary = """{
    "image": "${server.getMetadata().getName()}",
    "pixel_size_um": ${pixelSize},
    "total_cells": ${detections.size()},
    "positive_cells": ${positiveCount},
    "negative_cells": ${negativeCount},
    "positivity_pct": ${String.format("%.2f", positivityPct)},
    "dab_threshold": ${threshold}
}"""
new File(jsonPath).text = summary.trim()
println "JSON exported to: " + jsonPath

println "======================================"
println "PIPELINE FINISHED SUCCESSFULLY"
println "======================================"
