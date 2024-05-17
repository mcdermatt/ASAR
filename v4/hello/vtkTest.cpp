#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h> 
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkProperty.h>
#include <vtkCubeAxesActor2D.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkInteractorStyleTrackball.h>
#include <vtkSimplePointsReader.h>
#include <vtkWarpScalar.h>
#include <vtkAxisActor2D.h>


// int mains(int, char *[])
// {


// // Read the file
//   vtkSmartPointer<vtkSimplePointsReader> reader =vtkSmartPointer<vtkSimplePointsReader>::New();
//   reader->SetFileName ( "simple.xyz" );
//   reader->Update();

//   vtkSmartPointer<vtkPolyData> inputPolyData = vtkSmartPointer<vtkPolyData>::New();
//   inputPolyData ->CopyStructure(reader->GetOutput());


//   // warp plane
//   vtkSmartPointer<vtkWarpScalar> warp = vtkSmartPointer<vtkWarpScalar>::New();
//   warp->SetInput(inputPolyData);
//   warp->SetScaleFactor(0.0);

//   // Visualize
//   vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
//   mapper->SetInputConnection(warp->GetOutputPort());



//   vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
//   actor->GetProperty()->SetPointSize(4);
//   actor->SetMapper(mapper);

//   vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
//   vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
//   renderWindow->AddRenderer(renderer);
//   vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
//   renderWindowInteractor->SetRenderWindow(renderWindow);

//   renderer->AddActor(actor);
//   renderer->SetBackground(.3, .6, .3);
//   renderWindow->Render();

//   vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
//   renderWindowInteractor->SetInteractorStyle(style);

//   // add & render CubeAxes
//   vtkSmartPointer<vtkCubeAxesActor2D> axes = vtkSmartPointer<vtkCubeAxesActor2D>::New();
//   axes->SetInput(warp->GetOutput());
//   axes->SetFontFactor(3.0);
//   axes->SetFlyModeToNone();
//   axes->SetCamera(renderer->GetActiveCamera());

//   vtkSmartPointer<vtkAxisActor2D> xAxis = axes->GetXAxisActor2D();
//   xAxis->SetAdjustLabels(1);

//   renderer->AddViewProp(axes);
//   renderWindowInteractor->Start();

//   return EXIT_SUCCESS;
// }