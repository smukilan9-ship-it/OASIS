import qupath.ext.instanseg.core.InstanSeg
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.common.GeneralTools

def imageData = getCurrentImageData()
if (imageData == null) { println "ERROR: No image loaded"; return }
def server = imageData.getServer()

println "======================================"
println "STARTING ANALYSIS"
println "======================================"
println "Image: " + server.getMetadata().getName()
println "Width: " + server.getWidth()
println "Height: " + server.getHeight()

removeAllObjects()
setImageType('BRIGHTFIELD_H_DAB')

double pixelSize = 0.6060606060606061
setPixelSizeMicrons(pixelSize, pixelSize)
println "Pixel size: " + pixelSize + " um/px"

def roi = ROIs.createRectangleROI(0, 0, server.getWidth(), server.getHeight(), ImagePlane.getDefaultPlane())
def annotation = PathObjects.createAnnotationObject(roi)
addObject(annotation)
selectObjects(annotation)
println "Full image annotation created"

println "Running InstanSeg..."
def instanseg = InstanSeg.builder()
    .modelPath("/Users/mukilan/QuPath/v0.7/instanseg/downloaded/brightfield_nuclei-0.1.1")
    .device("mps")
    .nThreads(4)
    .tileDims(512)
    .interTilePadding(32)
    .makeMeasurements(true)
    .randomColors(false)
    .build()

instanseg.detectObjects()
println "InstanSeg completed"

def detections = getDetectionObjects()
println "Total cells detected: " + detections.size()
if (detections.isEmpty()) { println "WARNING: No detections found"; return }

double threshold = 0.1
println "DAB threshold: " + threshold + " (fixed OD)"

def positiveClass = getPathClass("Positive")
def negativeClass = getPathClass("Negative")
int positiveCount = 0
int negativeCount = 0

detections.each { cell ->
    def dab = cell.getMeasurementList().get("DAB: Mean")
    if (dab != null && !dab.isNaN() && dab > threshold) {
        cell.setPathClass(positiveClass); positiveCount++
    } else {
        cell.setPathClass(negativeClass); negativeCount++
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

def outDir = new File("/Users/mukilan/Desktop/ihc_spatial_results/LL477_CD8_x10_3__roi0")
if (!outDir.exists()) outDir.mkdirs()
def imageName = GeneralTools.stripExtension(server.getMetadata().getName())

def csvPath = new File(outDir, imageName + "_detections.csv").getAbsolutePath()
saveDetectionMeasurements(csvPath)
println "CSV exported to: " + csvPath

def jsonPath = new File(outDir, imageName + "_summary.json").getAbsolutePath()
def summary = """{
    "image": "${server.getMetadata().getName()}",
    "pixel_size_um": ${pixelSize},
    "image_width": ${server.getWidth()},
    "image_height": ${server.getHeight()},
    "total_cells": ${detections.size()},
    "positive_cells": ${positiveCount},
    "negative_cells": ${negativeCount},
    "positivity_pct": ${String.format("%.2f", positivityPct)},
    "dab_threshold": ${String.format("%.4f", threshold)}
}"""
new File(jsonPath).text = summary.trim()
println "JSON exported to: " + jsonPath

def geojsonPath = new File("/Users/mukilan/Desktop/ihc_spatial_results/LL477_CD8_x10_3__roi0", imageName + "_detections.geojson").getAbsolutePath()
exportObjectsToGeoJson(getDetectionObjects(), geojsonPath, "FEATURE_COLLECTION")
println "GeoJSON exported to: " + geojsonPath

println "======================================"
println "PIPELINE FINISHED SUCCESSFULLY"
println "======================================"
