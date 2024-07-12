gwl-modeling
============

Initialize the project and create a virtual environment
-------------------------------------------------------

1. Clone the repository

.. code:: bash

    git clone https://github.com/dfguerrerom/gwl-modeling.git

2. Change the config file `gee_scripts/config.toml.example` to `gee_scripts/config.toml` and fill in the :code:`ee-project` field with your Earth Engine project name.

.. code:: bash

    [config]
    ee-project="your_ee_project_name"


3. Create a virtual environment and install the dependencies

.. code:: bash

    cd gwl-modeling

    python3 gee_scripts/init_venv.py



