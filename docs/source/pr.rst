Submitting a Pull Request
==================================

We welcome contributions from the open source community. The main approach to communicate with and to make contribution to DeepFlame is to open a pull request. 

1. Fork the `DeepFlame repository <https://github.com/deepmodeling/deepflame-dev/tree/master>`_.
2. Pull your forked repository, and create a new git branchmake to your changes in it:

.. code-block:: bash
   
    git checkout -b my-fix-branch
    

3. Coding your patch

4. After tests passed, commit your changes with a proper message.

5. Push your branch to GitHub:

.. code-block:: bash
 
    git push origin my-fix-branch
    

6. In GitHub, send a pull request with ``deepmodeling/deepflame-dev`` as the base repository.

7. After your pull request is merged, you can safely delete your branch and sync the changes from the main (upstream) repository:

* Delete the remote branch on GitHub either through the GitHub web UI or your local shell as follows:

.. code-block:: bash
       
    git push origin --delete my-fix-branch
    

* Check out the master branch:

.. code-block:: bash
      
    git checkout develop -f


* Delete the local branch:

.. code-block:: bash
      
    git branch -D my-fix-branch


* Update your master with the latest upstream version:

.. code-block:: bash
       
    git pull --ff upstream develop

