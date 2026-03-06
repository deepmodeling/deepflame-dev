Foam::argList args(argc,argv,true,true,/*initialise=*/false);
if (!args.checkRootCase())
{
    Foam::FatalError.exit();
}