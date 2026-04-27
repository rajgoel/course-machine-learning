using HTTP, JSON3, Plots, Statistics, Dates, CSV, DataFrames, REPL.TerminalMenus


"""
    fetch_stock_data(symbol="AAPL", period="1y")

Fetch stock price data from Yahoo Finance API.

# Arguments
- `symbol::String`: Stock symbol (default: "AAPL")  
- `period::String`: Time period for data (default: "1y")
  Valid periods: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"

# Returns
- `Tuple{Vector{DateTime}, Vector{Float64}}`: (dates, closing_prices) or (nothing, nothing) if error

# Example
```julia
dates, prices = fetch_stock_data("MSFT", "6mo")
```
"""
function fetch_stock_data(symbol="AAPL", period="1y")
    # Use Yahoo Finance API (free, no key required)
    url = "https://query1.finance.yahoo.com/v8/finance/chart/$(symbol)?range=$(period)&interval=1d"
    
    try
        response = HTTP.get(url)
        data = JSON3.read(String(response.body))
        
        # Extract timestamps and closing prices
        timestamps = data.chart.result[1].timestamp
        prices = data.chart.result[1].indicators.quote[1].close
        
        # Convert timestamps to dates and filter out missing prices
        dates = [Dates.unix2datetime(ts) for ts in timestamps]
        valid_indices = findall(!isnothing, prices)
        
        return dates[valid_indices], Float64.(prices[valid_indices])
    catch e
        println("Error fetching data: $e")
        return nothing, nothing
    end
end

"""
    load_companies()

Load Dow 30 company data from CSV file.

# Returns
- `DataFrame`: Company data with Symbol and Name columns

# Example
```julia
companies = load_companies()
println(companies.Symbol[1])  # First company symbol
```
"""
function load_companies()
    csv_path = joinpath(@__DIR__, "dow30_companies.csv")
    return CSV.read(csv_path, DataFrame)
end

"""
    plot_stock_with_moving_average(symbol, window_size, period)

Plot stock price with moving average overlay.

# Arguments
- `symbol::String`: Stock symbol (e.g., "AAPL", "MSFT")
- `window_size::Int`: Moving average window size in days
- `period::String`: Time period for data (e.g., "1y", "6mo", "3mo")

# Returns
- `Plots.Plot`: Plot object that can be displayed or saved with savefig()

# Example
```julia
# Create plot
p = plot_stock_with_moving_average("AAPL", 20, "1y")

# Save plot
savefig(p, "apple_stock.png")
```
"""
function plot_stock_with_moving_average(symbol, window_size, period)
    dates, prices = fetch_stock_data(symbol, period)
    
    if isnothing(dates)
        return nothing
    end
    
    # Create a uniform averaging filter of length n: [1/n, 1/n, ...., 1/n]
    filter = ones(window_size) / window_size
    
    # Use the generic sliding filter function
    moving_average = sliding_filter(prices, filter; operation=sum)
    
    stock_plot = plot(dates, prices, label="$(symbol) Daily Close", linewidth=1, alpha=0.7, color=:blue)
    plot!(stock_plot, dates[window_size:end], moving_average, label="$(window_size)-Day Moving Average", linewidth=2, color=:red)
    
    title!(stock_plot, "$(symbol) Stock Price with Moving Average")
    xlabel!(stock_plot, "Date")
    ylabel!(stock_plot, "Price (USD)")
    
    return stock_plot
end

"""
    stock_data_moving_average()

Interactive stock data visualization with arrow key menus.

Provides an interactive interface to:
1. Select a company from Dow 30 companies
2. Choose time period (1mo, 3mo, 6mo, 1y, 2y)
3. Set moving average window (5, 10, 20, 30, 50 days)

# Returns
- `Plots.Plot`: Stock price plot with moving average overlay

# Usage
Navigate menus with arrow keys and Enter to select.

# Example
```julia
# Interactive demo
stock_data_moving_average()
```
"""
function stock_data_moving_average()
    companies = load_companies()
    
    # Company selection menu
    company_options = ["$(row.Symbol): $(row.Name)" for row in eachrow(companies)]
    company_menu = RadioMenu(company_options, pagesize=20)
    println("\n📈 Select a company (use ↑/↓ arrows, Enter to select):")
    company_choice = request(company_menu)
    
    selected_symbol = companies.Symbol[company_choice]
    
    # Time period selection menu
    period_options = [("1mo", "1 month"), ("3mo", "3 months"), ("6mo", "6 months"), 
                     ("1y", "1 year"), ("2y", "2 years")]
    period_display = [desc for (code, desc) in period_options]
    period_menu = RadioMenu(period_display)
    println("\n📅 Select time period:")
    period_choice = request(period_menu)
    
    selected_period = period_options[period_choice][1]
    
    # Moving average window selection
    window_options = [5, 10, 20, 30, 50]
    window_display = ["$(w) days" for w in window_options]
    window_menu = RadioMenu(window_display)
    println("\n📊 Select moving average window:")
    window_choice = request(window_menu)
    
    selected_window_size = window_options[window_choice]
    
    println("\nFetching data for $(selected_symbol)...")
    plot_stock_with_moving_average(selected_symbol, selected_window_size, selected_period)
end
