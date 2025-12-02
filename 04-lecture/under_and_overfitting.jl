using Plots, Random

  function true_fun(X)
      return cos.(1.5 * π * X)
  end

  # Set seed for reproducibility
  Random.seed!(0)

  n_samples = 30
  degrees = [1, 5, 21]
  titles = ["Low capacity (underfitting)","Moderate capacity (good fit)","High capacity (overfitting)"]

  X = sort(rand(n_samples))
  y = true_fun(X) + 0.1 * randn(n_samples)

  # Create polynomial design matrix
  function polynomial_features(x, degree)
      n = length(x)
      X = ones(n, degree + 1)  # Include bias term
      for d in 1:degree
          X[:, d + 1] = x .^ d
      end
      return X
  end

  # Fit polynomial regression
  function fit_polynomial(X, y, degree)
      X_poly = polynomial_features(X, degree)
      return X_poly \ y  # Linear least squares
  end

  # Predict with polynomial
  function predict_polynomial(X_new, coeffs, degree)
      X_poly = polynomial_features(X_new, degree)
      return X_poly * coeffs
  end

  # Create plots
  p = plot(layout=(1, 3), size=(1000, 300))

  X_test = range(0, 1, length=100)

  for (i, degree) in enumerate(degrees)
      # Fit model
      coeffs = fit_polynomial(X, y, degree)

      # Make predictions
      y_pred = predict_polynomial(X_test, coeffs, degree)

      # Plot
#      plot!(p[i], X_test, true_fun(X_test), label="True function", lw=2)
      plot!(p[i], X_test, y_pred, label="Model", lw=2, color=:firebrick)
      scatter!(p[i], X, y, label="Samples", ms=2, color=:gray, markerstroke=0)

      plot!(p[i], xlabel="x", ylabel="y", xlims=(0, 1), ylims=(-2, 2))
      plot!(p[i], title=titles[i], legend=:topright, titlefontsize=10 )
  end

  display(p)
  savefig(p, "under_and_overfitting.png")
