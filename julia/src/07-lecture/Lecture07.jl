"""
    Lecture07

Implementation of autoencoders

# Available Functions

- `demo()`: Train autoencoder for MNIST handwritten digits
- `display_latent_space(encoder, X_flat, Y_label)`: Display 2D latent space visualization
- `display_reconstruction(model, image)`: Display original vs reconstructed image

## Usage

```julia
using MachineLearningCourse
Lecture07.demo()
```

```julia
using MachineLearningCourse
X_flat, Y_label, autoencoder, encoder = Lecture07.demo(display=false)
i = 1 # set image index
Lecture07.display_reconstruction(autoencoder, X_flat[:, i])
```

```julia
using MachineLearningCourse
X_flat, Y_label, autoencoder, encoder = Lecture07.demo(display=false)
Lecture07.display_latent_space(encoder, X_flat, Y_label)
```

"""
module Lecture07

# Include implementation
include("AutoEncoder.jl")
include("Demo.jl")

export demo, display_latent_space, display_reconstruction

end # module Lecture07
