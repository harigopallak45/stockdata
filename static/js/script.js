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

                try {
                    const response = await fetch('/fetch_dataset', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    if (result.company_info) {
                        document.getElementById('company-info').innerHTML = `
                            <p><strong>Company Name:</strong> ${result.company_info.company_name}</p>
                            <p><strong>Description:</strong> ${result.company_info.description}</p>
                        `;
                    } else {
                        alert(result.error || 'Failed to fetch dataset');
                    }
                } catch (error) {
                    console.error('Error fetching dataset:', error);
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
            const result = await response.json();
            if (result.historical_data) {
                updateHistoricalGraph(result.historical_data);
                updateComparisonGraph(result.prediction_data);
                updateMonthWiseDiff(result.month_wise_diff);
            } else {
                alert(result.error || 'Failed to run analysis');
            }
        } catch (error) {
            console.error('Error running analysis:', error);
        }
    });

    // Function to update the historical graph
    function updateHistoricalGraph(data) {
        // Example implementation (replace with actual graph update logic)
        console.log('Updating historical graph with data:', data);
        document.getElementById('historical-graph').innerHTML = '<p>Historical graph will be displayed here.</p>';
    }

    // Function to update the comparison graph
    function updateComparisonGraph(data) {
        // Example implementation (replace with actual graph update logic)
        console.log('Updating comparison graph with data:', data);
        document.getElementById('comparison-graph').innerHTML = '<p>Comparison graph will be displayed here.</p>';
    }

    // Function to update month-wise data difference
    function updateMonthWiseDiff(data) {
        // Example implementation (replace with actual graph update logic)
        console.log('Updating month-wise difference with data:', data);
        document.getElementById('month-wise-diff').innerHTML = '<p>Month-wise data difference will be displayed here.</p>';
    }

    // Function to fetch and display live data
    async function fetchLiveData() {
        try {
            const response = await fetch('/live_data');
            const result = await response.json();
            document.getElementById('live-graph').innerHTML = `<p>Live Data: ${result.current_price}</p>`;
        } catch (error) {
            console.error('Error fetching live data:', error);
        }
    }

    // Call fetchLiveData periodically
    setInterval(fetchLiveData, 60000); // Update every minute
});
