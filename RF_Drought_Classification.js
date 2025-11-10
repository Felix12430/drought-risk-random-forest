// Load training points
// Load raster datasets as individual images 
// Rename bands to avoid duplication
var vci = ee.Image("projects/ee-felixkiruki1/assets/VCI_Export").rename("VCI");
var spi = ee.Image("projects/ee-felixkiruki1/assets/SPI3_Drought_2005_2010_2011_Clipped").rename("SPI3");
var tci = ee.Image("projects/ee-felixkiruki1/assets/TCI_Export").rename("TCI");
var lulc = ee.Image("projects/ee-felixkiruki1/assets/LULC_Export").rename("LULC");  
// Load MODIS NDVI dataset and clip to Marsabit boundary
var ndvi = ee.ImageCollection("MODIS/006/MOD13Q1")
             .filterDate("2000-01-01", "2024-12-31")
             .filterBounds(marsabit)  
             .select("NDVI")  
             .mean() 
             .multiply(0.0001)  // Scale MODIS NDVI values (0 to 1)
             .clip(marsabit) 
             .resample("bilinear")  
             .reproject({crs: "EPSG:4326", scale: 250});  // MODIS original resolution
//Load DEM
var dem = ee.Image("USGS/SRTMGL1_003").select("elevation");
// Clip DEM and NDVI to Marsabit County
var demClipped = dem.clip(marsabit);
var ndviClipped = ndvi.clip(marsabit);
// Define visualization parameters
var demVis = {
  min: 0,
  max: 3000,
  palette: ['blue', 'green', 'yellow', 'orange', 'red']
};
// Add NDVI and DEM Layer to Map
Map.centerObject(marsabit, 8);
Map.addLayer(ndviClipped, {min: 0, max: 1, palette: ['brown', 'yellow', 'green']}, "NDVI (MODIS)");
Map.addLayer(demClipped, demVis, "Clipped SRTM DEM");
// Combine auxiliary predictors
var predictors = ndvi.addBands(dem);
// Function to train downscaling model
function downscale(feature, featureName) {
  var trainingSamples = predictors.addBands(feature).sample({
  region: marsabit,  
  scale: 1000, 
  numPixels: 5000
});
  var model = ee.Classifier.smileRandomForest(100).train({
    features: trainingSamples,
    classProperty: featureName,
    inputProperties: predictors.bandNames()
  });
  return predictors.classify(model.setOutputMode('REGRESSION'));
}
// Downscale each feature separately
var downscaledVCI = downscale(vci, "VCI");
var downscaledSPI = downscale(spi, "SPI3");
var downscaledTCI = downscale(tci, "TCI");
var downscaledLULC = downscale(lulc, "LULC").round(); // LULC is categorical
// Stack all downscaled features into a single image
var stackedImage = downscaledVCI.addBands(downscaledSPI)
                   .addBands(downscaledTCI)
                   .addBands(downscaledLULC);
var classCounts = trainingPoints.aggregate_histogram('class');
print('Class Distribution:', classCounts);
// Extract raster values at training point locations
var trainingData = stackedImage.sampleRegions({
  collection: trainingPoints,
  properties: ['class'],  
  scale: 1000,
  geometries: true
});
// Split data into training (80%) and testing (20%)
var trainingSplit = 0.8;
var trainingSet = trainingData.randomColumn();
var trainData = trainingSet.filter(ee.Filter.lt('random', trainingSplit));
var testData = trainingSet.filter(ee.Filter.gte('random', trainingSplit));
print("Training Data Sample:", trainingData.first());
var featureBands = stackedImage.bandNames();
// Rename the bands to more meaningful names
var renamedImage = stackedImage.rename(["VCI", "SPI", "TCI", "LULC"]);
// Print the new band names
print("Renamed Feature Bands:", renamedImage.bandNames());
// Train Random Forest Classifier
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100, 
  minLeafPopulation: 3,
  bagFraction: 0.7
}).train({
  features: trainingData, 
  classProperty: 'class', 
  inputProperties: featureBands
});
// Get feature importance dictionary
var importance = ee.Dictionary(classifier.explain().get('importance'));
print('Raw Feature Importance:', importance);
// Get feature names (keys) and importance values
var keys = importance.keys();
var values = keys.map(function(k) { return importance.getNumber(k); });
// Compute total importance
var totalImportance = values.reduce(ee.Reducer.sum());
// Print results
print("Feature Names:", keys);
print("Feature Importance Values:", values);
print("Total Importance:", totalImportance);
// Normalize importance values to get percentage contribution
var normalizedImportance = values.map(function(v) {
  return ee.Number(v).divide(totalImportance).multiply(100);
});
// Combine feature names with their normalized importance values
var importanceList = keys.zip(normalizedImportance);
// Print the result
print("Normalized Feature Importance (%):", importanceList);
// Apply classification to the stacked image
var classified = stackedImage.classify(classifier);
// Evaluate model with test data
var confusionMatrix = classifier.confusionMatrix();
print("Confusion Matrix:", confusionMatrix);
print("Accuracy:", confusionMatrix.accuracy());
// Apply trained classifier to the test dataset
var validated = testData.classify(classifier);
// Compute the confusion matrix (error matrix)
var testAccuracy = validated.errorMatrix('class', 'classification');
print('Validation Error Matrix:', testAccuracy);
print('Validation Accuracy:', testAccuracy.accuracy());
// Export NDVI clipped
Export.image.toDrive({
  image: ndviClipped,
  description: "NDVI_Clipped_Marsabit",
  folder: "GEE_Exports", 
  scale: 250, 
  region: marsabit,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});
// Export DEM clipped
Export.image.toDrive({
  image: demClipped,
  description: "DEM_Clipped_Marsabit",
  folder: "GEE_Exports", 
  scale: 30, // SRTM DEM has a 30m resolution
  region: marsabit,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});
Export.image.toDrive({
  image: classifiedClipped,
  description: 'Drought_Risk_Map',
  folder: 'GEE_Exports',
  scale: 1000,  
  region: marsabit,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});
// Convert Confusion Matrix to FeatureCollection
var confusionData = ee.Feature(null, {
  'ConfusionMatrix': confusionMatrix.array(),
  'Accuracy': confusionMatrix.accuracy()
});
// Export Confusion Matrix & Accuracy
Export.table.toDrive({
  collection: ee.FeatureCollection([confusionData]),
  description: 'Model_Accuracy_Results',
  fileFormat: 'CSV'
});
Export.table.toDrive({
  collection: validated,
  description: 'Validation_Results',
  fileFormat: 'CSV'
});
var riskStats = smoothedMap.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: marsabit,
  scale: 1000,
  bestEffort: true
});
print("Drought Risk Distribution:", riskStats);
// Load the classified drought risk map from the Random Forest model
var droughtRiskMap = classifiedClipped;  // Ensure this is your RF classification output
// Define the visualization parameters
var visParams = {
  min: 0,
  max: 3,  // Adjust based on drought classes
  palette: ['blue', 'green', 'yellow', 'orange', 'red']  // Example drought severity colors
};
// Convert classification output to RGB for export
var visImage = droughtRiskMap.visualize(visParams);
// Create a legend
var legend = ui.Panel({
  style: {
    position: 'bottom-right',
    padding: '8px',
    backgroundColor: 'white'
  }
});
// Define legend labels
var legendLabels = [
  {label: 'Very high', color: 'blue'},
  {label: 'high', color: 'green'},
  {label: 'Moderate', color: 'yellow'},
  {label: 'low', color: 'orange'},
  {label: 'Very low', color: 'red'}
];
// Add legend items
legendLabels.forEach(function(item) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: item.color,
      padding: '10px',
      margin: '4px',
      color: 'white'
    }
  });
  var label = ui.Label(item.label, {margin: '4px'});
  legend.add(ui.Panel([colorBox, label], ui.Panel.Layout.Flow('horizontal')));
});
// Add title
var title = ui.Label({
  value: 'Drought Severity Map’,
  style: {
    fontSize: '16px',
    fontWeight: 'bold',
    margin: '10px',
    position: 'top-center'
  }
});
// Add legend & title to the map
ui.root.insert(0, title);
Map.add(legend);
// Export the visualized map to Google Drive
Export.image.toDrive({
  image: visImage,
  description: "Drought_Severity_Map2",
  folder: "GEE_Exports",
  fileNamePrefix: "Drought_Severity_Map2",
  scale: 1000,
  region: marsabit,
  maxPixels: 1e13,
  formatOptions: { cloudOptimized: true }
});
print("Drought Severity Map export initiated. Check Google Drive.");


