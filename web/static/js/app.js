// BROHKR Market Visualizer - Client-side JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const exchangeSelect = document.getElementById('exchange');
    const symbolInput = document.getElementById('symbol');
    const timeframeSelect = document.getElementById('timeframe');
    const refreshBtn = document.getElementById('refresh-btn');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // Chart containers
    const priceChartContainer = document.getElementById('price-chart-container');
    const orderBookContainer = document.getElementById('order-book-container');
    const exchangeComparisonContainer = document.getElementById('exchange-comparison-container');
    
    // Initialize the application
    init();
    
    // Event listeners
    refreshBtn.addEventListener('click', refreshData);
    
    // Tab switching
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Update active tab pane
            tabPanes.forEach(pane => pane.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            // Refresh the data for the active tab
            refreshData();
        });
    });
    
    // Initialize the application
    async function init() {
        try {
            // Fetch available exchanges
            const response = await fetch('/api/exchanges');
            const data = await response.json();
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Populate exchange selector
            data.exchanges.forEach(exchange => {
                const option = document.createElement('option');
                option.value = exchange;
                option.textContent = exchange;
                exchangeSelect.appendChild(option);
            });
            
            // Select first exchange by default
            if (data.exchanges.length > 0) {
                exchangeSelect.value = data.exchanges[0];
            }
            
            // Load initial data
            refreshData();
            
        } catch (error) {
            showError('Failed to initialize application: ' + error.message);
        }
    }
    
    // Refresh data based on active tab
    async function refreshData() {
        const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
        const exchange = exchangeSelect.value;
        const symbol = symbolInput.value;
        const timeframe = timeframeSelect.value;
        
        if (!exchange || !symbol) {
            showError('Please select an exchange and enter a symbol');
            return;
        }
        
        try {
            switch (activeTab) {
                case 'price-chart':
                    await loadPriceChart(exchange, symbol, timeframe);
                    break;
                case 'order-book':
                    await loadOrderBook(exchange, symbol);
                    break;
                case 'exchange-comparison':
                    await loadExchangeComparison(symbol);
                    break;
            }
        } catch (error) {
            showError('Failed to refresh data: ' + error.message);
        }
    }
    
    // Load price chart data
    async function loadPriceChart(exchange, symbol, timeframe) {
        showLoading(priceChartContainer);
        
        try {
            const response = await fetch(`/api/price_chart?exchange=${exchange}&symbol=${symbol}&timeframe=${timeframe}`);
            const data = await response.json();
            
            if (data.error) {
                showError(data.error, priceChartContainer);
                return;
            }
            
            // Parse the chart JSON and render it
            const chartData = JSON.parse(data.chart);
            Plotly.newPlot(priceChartContainer, chartData.data, chartData.layout);
            
        } catch (error) {
            showError('Failed to load price chart: ' + error.message, priceChartContainer);
        }
    }
    
    // Load order book data
    async function loadOrderBook(exchange, symbol) {
        showLoading(orderBookContainer);
        
        try {
            const response = await fetch(`/api/order_book?exchange=${exchange}&symbol=${symbol}`);
            const data = await response.json();
            
            if (data.error) {
                showError(data.error, orderBookContainer);
                return;
            }
            
            // Parse the chart JSON and render it
            const chartData = JSON.parse(data.chart);
            Plotly.newPlot(orderBookContainer, chartData.data, chartData.layout);
            
        } catch (error) {
            showError('Failed to load order book: ' + error.message, orderBookContainer);
        }
    }
    
    // Load exchange comparison data
    async function loadExchangeComparison(symbol) {
        showLoading(exchangeComparisonContainer);
        
        try {
            const response = await fetch(`/api/exchange_comparison?symbol=${symbol}`);
            const data = await response.json();
            
            if (data.error) {
                showError(data.error, exchangeComparisonContainer);
                return;
            }
            
            // Parse the chart JSON and render it
            const chartData = JSON.parse(data.chart);
            Plotly.newPlot(exchangeComparisonContainer, chartData.data, chartData.layout);
            
        } catch (error) {
            showError('Failed to load exchange comparison: ' + error.message, exchangeComparisonContainer);
        }
    }
    
    // Helper function to show loading indicator
    function showLoading(container) {
        container.innerHTML = '<div class="loading">Loading data...</div>';
    }
    
    // Helper function to show error message
    function showError(message, container = null) {
        const errorMessage = `<div class="error">${message}</div>`;
        
        if (container) {
            container.innerHTML = errorMessage;
        } else {
            // Show global error
            console.error(message);
        }
    }
});