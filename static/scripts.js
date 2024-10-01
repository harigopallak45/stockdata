document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');
    const downloadForm = document.getElementById('downloadForm');
    const resultsDiv = document.getElementById('results');
    const plotDiv = document.getElementById('plot');
    const companyInfoP = document.getElementById('companyInfo');
    const prophetPlotImage = document.getElementById('prophetPlotImage');
    const lstmPlotImage = document.getElementById('lstmPlotImage');
    const xgboostPlotImage = document.getElementById('xgboostPlotImage');
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

            // Set the plot images (historical data)
            if (data.historical_data_plot_url) {
                plotImage.src = data.historical_data_plot_url;
                plotImage.style.display = 'block';
            }

            // Fetch and display prediction plots automatically after file upload
            await fetchAndDisplayPlot('/prophet_prediction', prophetPlotImage);
            await fetchAndDisplayPlot('/lstm_prediction', lstmPlotImage);
            await fetchAndDisplayPlot('/xgboost_prediction', xgboostPlotImage);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the file.');
        }
    });

    // Function to fetch and display prediction plots
    async function fetchAndDisplayPlot(endpoint, plotImageElement) {
        try {
            const response = await fetch(endpoint, { method: 'POST' });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            plotImageElement.src = url;
            plotImageElement.style.display = 'block';

            plotDiv.style.display = 'block';

        } catch (error) {
            console.error(`Error fetching plot from ${endpoint}:`, error);
            alert(`An error occurred while fetching the plot from ${endpoint}.`);
        }
    }

    // Handle form submission for downloading stock data
    downloadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const ticker = document.getElementById('ticker').value;
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        try {
            const response = await fetch('/download_stock_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ ticker, start_date: startDate, end_date: endDate })
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const result = await response.json();
            alert('Stock data downloaded successfully!');

            // Optionally, trigger file upload or show stock data after download
            $('#downloadModal').modal('hide'); // Close the modal

        } catch (error) {
            console.error('Error downloading stock data:', error);
            alert('An error occurred while downloading stock data.');
        }
    });
});
