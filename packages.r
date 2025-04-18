# Set the library path to user-local library
user_lib <- "~/R/library"
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE)
}

# Install packages to the user-local library
install.packages(c(
  "purrr", "R.cache", "hms", "later", "promises", "cachem", "htmltools", 
  "sass", "jquerylib", "memoise", "sparsevctrs", "tidyr", "minqa", "RcppEigen", 
  "gtable", "systemfonts", "textshaping", "httr", "classInt", "s2", "units", 
  "callr", "processx", "readr", "httpuv", "pkgbuild", "bslib", "viridis", 
  "recipes", "lme4", "BradleyTerry2", "shiny", "rmarkdown", "testthat"
), lib = user_lib, dependencies = TRUE)

# Ensure R is aware of the new library
.libPaths(user_lib)
