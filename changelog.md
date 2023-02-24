# 0.1.2
- Fixed a bug when loading a cuda model

# 0.1.1
- Changed the name of the CGP class attribute from "m" to "inducing_points"
- Now the load method of the classes returns the class itself instead of initializing the model and the likelihood
- Fixed a bug where a tensor was not initialized in cuda if it is available

# 0.1.0
- First version of the package