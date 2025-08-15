interface IMLP<T, int In, int Out>
{
    associatedtype Parameters : IDifferentiablePtrType;
    associatedtype Configuration;

    property Parameters parameters { get; }
    property Configuration configuration { get; }
    property DifferentialPtrPair<Parameters> parametersDual { get; }
}