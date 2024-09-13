document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');
    const downloadForm = document.getElementById('downloadForm');
    const resultsDiv = document.getElementById('results');
    const plotDiv = document.getElementById('plot');
    const companyInfoP = document.getElementById('companyInfo');
    const historicalDataP = document.getElementById('historicalData');
    const predictionDataP = document.getElementById('predictionData');
    const monthWiseDiffP = document.getElementById('monthWiseDiff');
    const liveDataP = document.getElementById('liveData');
    const plotImage = document.getElementById('plotImage');
    const monthWiseDiffPlotImage = document.getElementById('monthWiseDiffPlotImage');
    const fetchLiveDataBtn = document.getElementById('fetchLiveData');
    const showPlotBtn = document.getElementById('showPlot');
    const downloadBtn = document.getElementById('downloadBtn');

    // Initially hide the results and plot sections
    resultsDiv.style.display = 'none';
    plotDiv.style.display = 'none';

    // Show download form as a modal when the "Download" button is clicked
    downloadBtn.addEventListener('click', () => {
        $('#downloadModal').modal('show');
    });

    // Handle form submission for file upload
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Post the uploaded file to the backend
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }

            // Show results and plot sections after successful upload
            resultsDiv.style.display = 'block';
            plotDiv.style.display = 'block';

            // Display company info
            companyInfoP.textContent = `Company Name: ${data.company_info.company_name}\nDescription: ${data.company_info.description}`;

            // Display live data (if available)
            if (data.live_data && !data.live_data.error) {
                liveDataP.textContent = JSON.stringify(data.live_data, null, 2);
            } else {
                liveDataP.textContent = 'No live data available.';
            }

            // Set the plot images (historical data and month-wise difference)
            plotImage.src = data.historical_data_plot_url;
            monthWiseDiffPlotImage.src = data.month_wise_diff_plot_url;

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the file.');
        }
    });

    // Handle the button click to fetch live data
    fetchLiveDataBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/live_data');

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const liveData = await response.json();
            if (liveData.error) {
                alert(`Error: ${liveData.error}`);
                return;
            }

            // Display fetched live data
            liveDataP.textContent = JSON.stringify(liveData, null, 2);
            resultsDiv.style.display = 'block';  // Show results section if it was hidden

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while fetching live data.');
        }
    });

    // Handle the button click to display the plot
    showPlotBtn.addEventListener('click', () => {
        if (plotImage.src || monthWiseDiffPlotImage.src) {
            plotDiv.style.display = 'block';
        } else {
            alert('No plots available to display.');
        }
    });

    // Handle form submission for downloading stock data
    downloadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const ticker = document.getElementById('ticker').value;
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        try {
            const response = await fetch(`/download_stock_data?ticker=${ticker}&start_date=${startDate}&end_date=${endDate}`);

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const result = await response.json();
            alert('Stock data downloaded successfully!');

            // Optionally, trigger file upload or show stock data after download
            // $('#downloadModal').modal('hide'); // Close the modal
            // Process downloaded data, like showing plots etc.
        } catch (error) {
            console.error('Error downloading stock data:', error);
            alert('An error occurred while downloading stock data.');
        }
    });
});
