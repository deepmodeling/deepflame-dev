/* For DeepFlame_Academic */
argList::addBoolOption("compareResults", "Set option [compareResults] to compare GPU results with OpenFOAM.");

Foam::argList args(argc,argv,true,true,/*initialise=*/false);
if (!args.checkRootCase())
{
    Foam::FatalError.exit();
}

bool compareResults = false;
if (args.optionFound("compareResults"))
{
    compareResults = true;
    Info << "SET [compareResults]" << endl;
}