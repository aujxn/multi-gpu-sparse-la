//                    DG Diffusion Test for Multi-GPU SpMM
//
// Compile with: make dg_diffusion
//
// Description:  This example creates a discontinuous Galerkin discretization of
//               the Poisson problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. It extracts the parallel matrix components
//               from MFEM/Hypre and tests them against our multi-GPU sparse
//               matrix-vector multiply library.
//
//               Based on MFEM example 14p (ex14p.cpp).

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include "_hypre_parcsr_mv.h"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 1.1 GPU initialization not required for size logging only

   // 2. Parse command-line options.
   const char *mesh_file = "mfem-4.8/data/star.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 3;
   real_t sigma = -1.0;
   real_t kappa = -1.0;
   bool verification = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive.");
   args.AddOption(&verification, "-verify", "--verification", "-no-verify",
                  "--no-verification", "Enable verification against Hypre.");
   args.Parse();
   
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   // 3. Read the mesh and set up parallel mesh
   Mesh mesh(mesh_file);
   int dim = mesh.Dimension();

   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // 4. Define DG finite element space
   DG_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec /* scalar DG space */);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   
   Array<int> dofs;
   fespace.GetElementDofs(0, dofs); // scalar DG case
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << size << endl;
      cout << "Element block size: " << dofs.Size() << endl;
   }

   // 5. Set up the bilinear form (matrix) and linear form (RHS)
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(zero, one, sigma, kappa));
   b.Assemble();

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a.Assemble();
   a.Finalize();

   // 6. Extract the parallel matrix
   OperatorHandle A;
   A.SetType(Operator::Hypre_ParCSR);
   a.ParallelAssemble(A);
   HypreParMatrix* hypre_A = A.As<HypreParMatrix>();

   // 7. Create test vectors
   ParGridFunction x(&fespace), y_hypre(&fespace), y_ours(&fespace);
   x.Randomize(1234);  // Set random test vector

   // 8. Compute reference SpMV with Hypre and log basic size info per rank
   hypre_A->Mult(x, y_hypre);

   // 10. Solve the system with Hypre for reference
   HypreBoomerAMG amg(*hypre_A);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(*hypre_A);
   cg.SetPreconditioner(amg);
   
   ParGridFunction solution(&fespace);
   solution = 0.0;
   cg.Mult(b, solution);

   // TODO: make parBELL and benchmark
   if (Mpi::Root())
   {
      cout << "DG diffusion example completed successfully!" << endl;
   }

   return 0;
}
