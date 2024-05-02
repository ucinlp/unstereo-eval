$(document).ready(function() {
    // Define chart variable
    let chart = null;
  
    // Function to load data from CSV file
    function loadDataFromJSON(callback) {
        d3.json("https://raw.githubusercontent.com/ucinlp/unstereo-eval/main/docs/assets/data/all_datasets.json")
        .then(data => {
            console.log(data);
            callback(data);
            
        });
    }
  
    function filterAndVisualizeData(dataset) {
      const datasetMenu = document.getElementById('dataset-menu');
      const constraintMenu = document.getElementById('eta-menu');
      const metricMenu = document.getElementById('metric-menu');
  
      // Event listener for dropdown menu changes
      datasetMenu.addEventListener('change', updateVisualization);
      constraintMenu.addEventListener('change', updateVisualization);
      metricMenu.addEventListener('change', updateVisualization);

      // Initial visualization
      updateVisualization();
  
      // Function to update the visualization based on the selected values
      function updateVisualization() {
        const selectedDataset = datasetMenu.value;
        const selectedConstraint = constraintMenu.value;
        const selectedMetric = metricMenu.value;
  
        // Filter the dataset based on the selected values
        const metricsData = dataset[selectedDataset][selectedConstraint]
        if (Object.keys(metricsData).length > 0) {
            const benchmarkSize = metricsData["num_examples"];
  
              // Clear previous results
            $('#benchmark-size').empty();
            $('#benchmark-size').append(benchmarkSize);
      
  
            // Clear existing chart if any
            d3.select('#chart').html('');
        };
        }
    }
    // Call the loadDataFromJSON function and pass the filterAndVisualizeData function as the callback
    loadDataFromJSON(filterAndVisualizeData);

    });