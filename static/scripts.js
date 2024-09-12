document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');
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

            // Show results and plot sections
            resultsDiv.style.display = 'block';
            plotDiv.style.display = 'block';

            // Display company info
            companyInfoP.textContent = `Company Name: ${data.company_info.company_name}\nDescription: ${data.company_info.description}`;

            // Display live data
            liveDataP.textContent = JSON.stringify(data.live_data, null, 2);

            // Set the plot image source to the URL of the generated plots
            plotImage.src = data.historical_data_plot_url;
            monthWiseDiffPlotImage.src = data.month_wise_diff_plot_url;

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the file.');
        }
    });

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

            // Display live data
            liveDataP.textContent = JSON.stringify(liveData, null, 2);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while fetching live data.');
        }
    });

    showPlotBtn.addEventListener('click', () => {
        plotDiv.style.display = 'block';
    });
});
