/*
install required packages:
sudo apt-get update
sudo apt-get install freeglut3 freeglut3-dev binutils-gold g++ cmake libglew-dev mesa-common-dev build-essential libglew1.5-dev libglm-dev

run opengl screen:
g++ screen.cpp utils.cpp -o screen -lglut -lGLU -lGL
*/

// Author: Mario Talevski
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include "parameter.h"
#include "utils.h"
#include "gameOfLife.h"
#include <GL/glut.h>  // GLUT, include glu.h and gl.h

#if defined(OS_WIN)
  #include <windows.h> // Use for windows
#endif

void display();
void idle();

nt windowsWidth = 1800;
int windowsHeight = (int) (windowsWidth * float(arrayHeight) / float(arrayWidth));
float offsetWidth = (1.8 / 1.15) / (float) (arrayWidth);
float offsetHeight = (1.8 / 1.15) / (float) (arrayHeight);

bool* gridOne = malloc_host<bool>(arrayHeight*arrayWidth, false);
bool* gridTwo = malloc_host<bool>(arrayHeight*arrayWidth, false);
bool* gridAns = malloc_host<bool>(arrayHeight*arrayWidth, false);

void updateSerial(bool* &gridOne, bool* &gridTwo){
    std::swap(gridOne, gridTwo);
    for(int a = 1; a <= gridHeight; a++)
    {
        for(int b = 1; b <= gridWidth; b++)
        {
            int alive =   gridTwo[(a-1)*arrayWidth + b-1]   + gridTwo[a*arrayWidth + b-1] + gridTwo[(a+1)*arrayWidth + b-1]
                        + gridTwo[(a-1)*arrayWidth + b]                                  + gridTwo[(a+1)*arrayWidth + b]
                        + gridTwo[(a-1)*arrayWidth + b+1]   + gridTwo[a*arrayWidth + b+1] + gridTwo[(a+1)*arrayWidth + b+1];;
            gridOne[a*arrayWidth + b] = ((alive == 3) || (alive==2 && gridTwo[a*arrayWidth + b]))?1:0; 
        }
    }
}

int main(int argc, char** argv) {

	char mode;
    cout << "Select Initialization mode, read from file or random sample (r/s): ";
    cin >> mode;

    initGrid(mode, gridOne);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(windowsWidth, windowsHeight);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Parallel Game of Life");
    glutDisplayFunc(display);
	glutIdleFunc(idle);
    glutMainLoop();
    return 0;
}

void display() {
   	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black and opaque
   	glClear(GL_COLOR_BUFFER_BIT);         // Clear the color buffer (background)
 
   	// Draw a Red 1x1 Square centered at origin
   	glBegin(GL_QUADS);              // Each set of 4 vertices form a quad
    	glColor3f(1.0f, 1.0f, 1.0f); // White

		float x_t = -0.9f, y_t = -0.9f;
		for(int a = 0; a <= gridHeight+1; a++)
		{
			x_t = -0.9f;
			for(int b = 0; b <= gridWidth+1; b++)
			{
				if (gridOne[a*arrayWidth+b] == true)
				{
					glVertex2f(x_t - offsetWidth/2, y_t + offsetHeight/2); 
					glVertex2f(x_t + offsetWidth/2, y_t + offsetHeight/2);
					glVertex2f(x_t + offsetWidth/2, y_t - offsetHeight/2);
					glVertex2f(x_t - offsetWidth/2, y_t - offsetHeight/2);
				}
				x_t += offsetWidth * 1.15;
			}
			y_t += offsetHeight * 1.15;
		}


   	glEnd();

	glutSwapBuffers();
}

void idle() {
	updateSerial(gridOne, gridTwo);
	glutPostRedisplay();
}
