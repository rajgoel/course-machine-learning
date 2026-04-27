using Printf

# Function to convert a number into an alphabetical letter (a, b, c, ...)
function get_letter_suffix(count)
    return Char(96 + count)  # 96 is the ASCII value before 'a'
end

function read_data(file_path)
    digits_data = []
    open(file_path, "r") do f
        while !eof(f)
            # Read 5 rows of 5 values
            grid = []
            for _ in 1:5
                push!(grid, split(readline(f)) |> x -> parse.(Int, x))
            end
            digit = parse(Int, readline(f))
            push!(digits_data, (grid, digit))
        end
    end
    return digits_data
end

function generate_svg(grid, digit, output_folder, suffix)
    width = 100
    height = 100
    cell_size = 20  # 20x20 pixels per grid cell

    svg_content = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $width $height" width="$width" height="$height">
    """

    for i in 1:5, j in 1:5
        x = (j - 1) * cell_size
        y = (i - 1) * cell_size
        color = if grid[i][j] == 1 "white" else "black" end
        svg_content *= "<rect x=\"$x\" y=\"$y\" width=\"$cell_size\" height=\"$cell_size\" fill=\"$color\" stroke=\"gray\" stroke-width=\"1\" />\n"
    end

    svg_content *= "</svg>"

    # Use the suffix (a, b, c, ...) to disambiguate file names
    output_path = joinpath(output_folder, "digit_$digit$suffix.svg")
    open(output_path, "w") do file
        write(file, svg_content)
    end
    println("Generated: $output_path")
end

function main(file_path, output_folder)
    data = read_data(file_path)
    digit_counts = Dict{Int, Int}()

    for (grid, digit) in data
        # Increment the count for the current digit
        if haskey(digit_counts, digit)
            digit_counts[digit] += 1
        else
            digit_counts[digit] = 1
        end

        # Get the letter suffix (a, b, c, ...) based on the count
        suffix = get_letter_suffix(digit_counts[digit])

        # Generate the SVG with the disambiguated file name
        generate_svg(grid, digit, output_folder, suffix)
    end
end

# Example usage:
file_path = "5x5digits.txt"  # Replace with your file path
output_folder = "svg"
# Create output directory if it doesn't exist
if !isdir(output_folder)
  mkdir(output_folder)
end
main(file_path, output_folder)

