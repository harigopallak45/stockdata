document.addEventListener('DOMContentLoaded', () => {
    const fetchDatasetBtn = document.getElementById('fetch-dataset-btn');
    const runAnalysisBtn = document.getElementById('run-analysis-btn');

    fetchDatasetBtn.addEventListener('click', () => {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.csv';
        fileInput.onchange = async () => {
            const file = fileInput.files[0];
            if (file) {
                const formData = new FormData();
                formData.append('file', file);
                
                // Display the name of the selected dataset immediately
                document.getElementById('selected-dataset').textContent = file.name;
                
                try {
                    const response = await fetch('/fetch_dataset', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    
                    if (result.company_info) {
                        document.getElementById('company-name').textContent = result.company_info.company_name || 'Not available';
                        document.getElementById('description').textContent = result.company_info.description || 'Not available';
                        runAnalysisBtn.disabled = false; // Enable the "Run Analysis" button
                    } else {
                        document.getElementById('company-name').textContent = 'Not available';
                        document.getElementById('description').textContent = 'Not available';
                        runAnalysisBtn.disabled = true; // Disable the "Run Analysis" button
                        alert(result.error || 'Failed to fetch dataset');
                    }
                } catch (error) {
                    console.error('Error fetching dataset:', error);
                    document.getElementById('company-name').textContent = 'Not available';
                    document.getElementById('description').textContent = 'Not available';
                    runAnalysisBtn.disabled = true; // Disable the "Run Analysis" button
                    alert('An error occurred while fetching the dataset.');
                }
            }
        };
        fileInput.click();
    });

    runAnalysisBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/run_analysis', {
                method: 'POST'
            });

            if (!response.ok) {
                const error = await response.json();
                alert(error.error || 'Failed to run analysis');
                return;
            }

            const result = await response.json();
            
            if (result.historical_data && result.prediction_data && result.month_wise_diff) {
                updateHistoricalGraph(result.historical_data);
                updateComparisonGraph(result.prediction_data);
                updateMonthWiseDiff(result.month_wise_diff);
            } else {
                alert(result.error || 'Failed to run analysis');
            }
        } catch (error) {
            console.error('Error running analysis:', error);
            alert('An unexpected error occurred. Check the console for details.');
        }
    });

    // Function to update the historical graph
    function updateHistoricalGraph(data) {
        // Example implementation (replace with actual graph update logic)
        console.log('Updating historical graph with data:', data);
        // Here you would use a library like Chart.js or D3.js to update the graph
        document.getElementById('historical-graph').innerHTML = '<p>Historical graph will be displayed here.</p>';
    }

    // Function to update the comparison graph
    function updateComparisonGraph(data) {
        // Example implementation (replace with actual graph update logic)
        console.log('Updating comparison graph with data:', data);
        // Here you would use a library like Chart.js or D3.js to update the graph
        document.getElementById('comparison-graph').innerHTML = '<p>Comparison graph will be displayed here.</p>';
    }

    // Function to update month-wise data difference
    function updateMonthWiseDiff(data) {
        // Example implementation (replace with actual graph update logic)
        console.log('Updating month-wise difference with data:', data);
        // Here you would use a library like Chart.js or D3.js to update the graph
        document.getElementById('month-wise-diff').innerHTML = '<p>Month-wise data difference will be displayed here.</p>';
    }

    // Function to fetch and display live data
    async function fetchLiveData() {
        try {
            const response = await fetch('/live_data');
            const result = await response.json();
            if (result.timestamp && result.price) {
                document.getElementById('live-graph').innerHTML = `<p>Live Data: ${result.price}</p>`;
            } else {
                document.getElementById('live-graph').innerHTML = '<p>No live data available.</p>';
            }
        } catch (error) {
            console.error('Error fetching live data:', error);
            document.getElementById('live-graph').innerHTML = '<p>Error fetching live data.</p>';
        }
    }

    // Call fetchLiveData periodically
    setInterval(fetchLiveData, 60000); // Update every minute
});
