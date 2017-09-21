
.. blogpost::
    :title: Call C# from Python
    :keywords: reference, blog, post
    :date: 2017-09-17
    :categories: C#, DLL

    A couple of questions must be answers to do that.

    * **Call C# from Python**: it is possible to use
      `Pythonnet <https://github.com/pythonnet/pythonnet>`_ or it is possible
      to create a DLL from C# which can be called from C:
      `Can you call a C# DLL from a C DLL? <https://stackoverflow.com/questions/728325/can-you-call-a-c-sharp-dll-from-a-c-dll>`_.
    * **Call C DLL from Python**: :epkg:`Python` has a module for that,
      `ctypes <https://docs.python.org/3/library/ctypes.html#loading-dynamic-link-libraries>`_.
    * **Compile C# on Linux**: because we want to create a portable code,
      see `How to compile c# on ubuntu <https://gist.github.com/lzomedia/4ce0da2c405c43b4341aef2f39cbfb84>`_

    If possible, it would be great to have a automated way to create
    files *.sln* and *.csproj*. The rest is coming soon.
