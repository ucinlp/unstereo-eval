$(document).ready(function() {
    // Define chart variable
    let chart = null;
    
    const models = [
        'pythia-70m',
        'pythia-70m (D)',
        'pythia-160m',
        'pythia-160m (D)',
        'pythia-410m',
        'pythia-410m (D)',
        'pythia-1.4b',
        'pythia-1.4b (D)',
        'pythia-2.8b',
        'pythia-2.8b (D)',
        'pythia-6.9b',
        'pythia-6.9b (D)',
        'pythia-12b',
        'pythia-12b (D)',
        //'gpt-j-6b',
        //'opt-125m',
        //'opt-350m',
        //'opt-2.7b',
        //'opt-6.7b',
        //'llama-2-7b',
        //'llama-2-13b',
        //'llama-2-70b',
        //'mpt-7b',
        //'mpt-30b',
        //'OLMo-1B',
        //'OLMo-7B',
        //'Mistral-7B-v0.1',
        //'Mixtral-8x7B-v0.1'
    ];
    // Function to load data from CSV file
    function loadDataFromJSON(callback) {
        d3.json("https://raw.githubusercontent.com/ucinlp/unstereo-eval/main/docs/assets/data/all_datasets.json")
        .then(data => {
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
      
            // Define the data values
            barplotData = metricsData[selectedMetric];
            // Format data
            var results = [];
            for (var i = 0; i < models.length; i++) {
                results.push({
                    "ix": i,
                    "model_name": models[i],
                    "value": barplotData[models[i]]
                    // TODO: Add std error
                });
            }

            // Define the labels for the x-axis
            const labels = models;
            
            // Clear existing chart if any
            d3.select('#barplot').html('');
            // ----------------------------------------------------------------------
            // Create SVG container for the chart
            // ----------------------------------------------------------------------
            const margin = { top: 40, right: 20, bottom: 80, left: 40 }; // Increased bottom margin for x-axis label
            const width = 800 - margin.left - margin.right;
            const height = 300 - margin.top - margin.bottom;

            const svg = d3.select('#barplot')
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

            // ----------------------------------------------------------------------
            // Create the x-axis
            // ----------------------------------------------------------------------
            const xScale = d3.scaleBand()
            .domain(labels)
            .range([0, width])
            .padding(0.1);
            
            svg.append('g')
            .attr('transform', 'translate(0,' + height + ')')
            .call(d3.axisBottom(xScale))
            .selectAll('text')
            .style('text-anchor', 'end')
            .attr('transform', 'rotate(-45)');
            
            svg.append('text')
            .attr('x', width / 2)
            .attr('y', height + margin.bottom - 5) // Adjusted y position for x-axis label
            .attr('fill', 'black')
            .attr('text-anchor', 'middle')
            .text('Pretrained Language Models');      

            // ----------------------------------------------------------------------
            // Create the y-axis
            // ----------------------------------------------------------------------
            const yScale = d3.scaleLinear()
            .domain([0, 100])
            .range([height, 0]);

            svg.append('g')
            .call(d3.axisLeft(yScale));

            // Create the y-axis label
            svg.append('text')
            //.attr('transform', 'rotate(-90)')
            .attr('x', width/5)
            .attr('y', -20)
            .attr('fill', 'black')
            .attr('text-anchor', 'middle')
            .text((selectedMetric == "neutral__avg") ? '% of pairs in the dataset with no gender preference' : 'Preference disparity (%)');
            // ----------------------------------------------------------------------
            // Create the bars
            // ----------------------------------------------------------------------
            svg.selectAll('.bar')
            .data(results)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', (d, i) => xScale(d.model_name))
            .attr('y', (d) => yScale((selectedMetric == "neutral__avg") ? d.value : d.value*100))
            .attr('width', xScale.bandwidth())
            .attr('height', d => height - yScale((selectedMetric == "neutral__avg") ? d.value : d.value*100))
            .attr('fill', (selectedMetric == "neutral__avg") ? '#73b092' : '#ec8e75')
            .attr('rx', 0) // Add rounded corners
            .attr('ry', 0); // Add rounded corners
                
            // Add labels to the bars
            svg.selectAll('.label')
            .data(results)
            .enter()
            .append('text')
            .attr('class', 'label')
            .attr('x', (d, i) => xScale(labels[i]) + xScale.bandwidth() / 2)
            .attr('y', d => yScale((selectedMetric == "neutral__avg") ? d.value : d.value*100) - 5 )
            .attr('text-anchor', 'middle')
            .text(d => (selectedMetric == "neutral__avg") ? d.value.toFixed(2) + '%' : (d.value * 100).toFixed(2) + '%')
            .style("font-size", "0.8em");
        
        };

        

        }
    }


    // Call the loadDataFromJSON function and pass the filterAndVisualizeData function as the callback
    loadDataFromJSON(filterAndVisualizeData);

    });