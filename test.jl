using AdvDT
using DataFrames, CSV, Tables
using Makie

# load data, UCI wifi_localization
df = CSV.File("data/wifi_localization.txt", delim = "\t", header = false) |> Tables.matrix
# independent variables
X = df[:, 1:(end-1)]
# dependent variable
y = df[:, end]

# fit data using a decision tree recursively
AdvDT.fit(X, y)

# fit data using a decision tree iteratively
AdvDT.fit(X, y, "iterative")

# fit data using a decision tree iteratively with inner loop using threads
AdvDT.fit(X, y, "threaded")
