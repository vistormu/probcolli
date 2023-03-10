# 0.1.4
- Migrated the log to vclog
- Fixed a bug where the load method did just not work
- Changed the name of the load method to "from_model" to emphasize that it is a constructor method

# 0.1.3
- Now a model can predict from a single value
- The training percentage is now more accurate

# 0.1.2
- Fixed a bug when loading a cuda model

# 0.1.1
- Changed the name of the CGP class attribute from "m" to "inducing_points"
- Now the load method of the classes returns the class itself instead of initializing the model and the likelihood
- Fixed a bug where a tensor was not initialized in cuda if it is available

# 0.1.0
- First version of the package