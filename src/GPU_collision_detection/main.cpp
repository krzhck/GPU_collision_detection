#include<GL/glut.h>
#include<iostream>
#include<math.h>
#include<windows.h>
#include "point.hpp"
#include "ball.hpp"
#include "light.hpp"
#include "camera.hpp"
#include "board.hpp"
#include "balllist.hpp"
#include "collision.cuh"
using namespace std;


//ȫ�ֳ���
const int WindowSizeX = 800, WindowSizeY = 600, WindowPlaceX = 100, WindowPlaceY = 100;
const char WindowName[] = "collision detection";
const float TimeOnce = 0.02; //ˢ��ʱ��
const int BallNum = 5;
const float XRange = 10, ZRange = 10, Height = 20, MaxRadius = 1; //������X,Y,Z��Χ��-X,X),(0,H),(-Z,Z)

//���գ����
Camera TheCamera(20.0f, 10.0f);
Light TheLight;

//����
Board Boards[6]; //�߽�

BallList Balls(XRange, Height, ZRange, BallNum, MaxRadius, TimeOnce);


//��ʼ����������
//��ʼ������
void InitWindow()
{
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WindowSizeX, WindowSizeY);
	glutInitWindowPosition(WindowPlaceX, WindowPlaceY);
	glutCreateWindow(WindowName);
	const GLubyte* OpenGLVersion = glGetString(GL_VERSION);
	const GLubyte* gluVersion = gluGetString(GLU_VERSION);
	printf("OpenGLʵ�ֵİ汾�ţ�%s\n", OpenGLVersion);
	printf("OGLU���߿�汾��%s\n", gluVersion);
	int dev = 0;
	cudaDeviceProp devProp;
	if (cudaGetDeviceProperties(&devProp, dev) == cudaSuccess)
	{
		std::cout << "ʹ��GPU device " << dev << ": " << devProp.name << std::endl;
		std::cout << "SM��������" << devProp.multiProcessorCount << std::endl;
		std::cout << "ÿ���߳̿�Ĺ����ڴ��С��" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
		std::cout << "ÿ���߳̿������߳�����" << devProp.maxThreadsPerBlock << std::endl;
		std::cout << "ÿ��EM������߳�����" << devProp.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "ÿ��EM������߳�������" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
	}
}

//��ʼ������
void InitLight()
{
	//������ɫģʽ
	glShadeModel(GL_SMOOTH);
	//���ó�ʼ����ɫ�������ɫ�������Ȼ���
	glClearColor(TheLight.Color[0], TheLight.Color[1], TheLight.Color[2], TheLight.Color[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//���ù�Դ��Ϣ
	glLightfv(GL_LIGHT0, GL_AMBIENT, TheLight.Ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, TheLight.Diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, TheLight.Specular);
	glLightfv(GL_LIGHT0, GL_POSITION, TheLight.Position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	//������ȼ�⣬��ֻ������ǰ���һ��
	glEnable(GL_DEPTH_TEST);
}

//��ʼ���߽�͵ذ�
void InitBoards()
{
	//8����
	Point DownA(-XRange, 0, -ZRange);
	Point DownB(-XRange, 0, ZRange);
	Point DownC(XRange, 0, -ZRange);
	Point DownD(XRange, 0, ZRange);
	Point UpA(-XRange, Height, -ZRange);
	Point UpB(-XRange, Height, ZRange);
	Point UpC(XRange, Height, -ZRange);
	Point UpD(XRange, Height, ZRange);

	//���õذ�͵���λ��
	Boards[0].InitPlace(DownA, DownB, DownD, DownC); // bottom
	Boards[1].InitPlace(DownA, DownB, UpB, UpA);
	Boards[3].InitPlace(DownC, DownD, UpD, UpC);
	Boards[2].InitPlace(DownA, DownC, UpC, UpA);
	Boards[4].InitPlace(DownB, DownD, UpD, UpB);
	Boards[5].InitPlace(UpA, UpB, UpD, UpC); // top

	GLfloat color_bottom[4] = { 1.0, 1.0, 1.0 , 1};
	GLfloat ambient_bottom[4] = { 0.4, 0.4, 0.4 , 1};
	GLfloat diffuse_bottom[4] = { 0.4, 0.4, 0.4 , 1};
	GLfloat specular_bottom[4] = { 0.2, 0.2, 0.2 , 1};
	GLfloat shininess_bottom = 20;

	Shader shader_bottom(color_bottom, ambient_bottom, diffuse_bottom, specular_bottom, shininess_bottom);
	Boards[0].InitColor(shader_bottom);

	//�������ܵ������
	GLfloat color_border[3] = { 1.0, 1.0, 1.0 };
	GLfloat ambient_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat diffuse_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat specular_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat shininess_border = 40;
	Shader shader_border(color_border, ambient_border, diffuse_border, specular_border, shininess_border);
	for (int i = 1; i < 5; i++)
	{
		Boards[i].InitColor(shader_border);
	}
}




//��ʼ����������
void InitScene()
{

	InitLight();
	InitBoards();
	Balls.InitBalls();
}

//���ƺ�������
//�������λ��
void SetCamera()
{
	glLoadIdentity();
	Point camera_place = TheCamera.CurrentPlace;//������ӵ������  
	Point camera_center = TheCamera.LookCenter;//�����ӵ���������
	gluLookAt(camera_place.x, camera_place.y, camera_place.z, camera_center.x, camera_center.y, camera_center.z, 0, 1, 0); //���ӵ㿴Զ��,y�᷽��(0,1,0)���Ϸ���  
}

//����2�߽�͵ذ�
void DrawBoards()
{
	for (int i = 0; i < 3; i++)
	{
		glColor3f(Boards[i].WallShader.Color[0], Boards[i].WallShader.Color[1], Boards[i].WallShader.Color[2]);
		glMaterialfv(GL_FRONT, GL_AMBIENT, Boards[i].WallShader.Ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, Boards[i].WallShader.Diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, Boards[i].WallShader.Specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, Boards[i].WallShader.Shininess);


		glBegin(GL_POLYGON);
		glVertex3f(Boards[i].PointList[0].x, Boards[i].PointList[0].y, Boards[i].PointList[0].z);
		glVertex3f(Boards[i].PointList[1].x, Boards[i].PointList[1].y, Boards[i].PointList[1].z);
		glVertex3f(Boards[i].PointList[2].x, Boards[i].PointList[2].y, Boards[i].PointList[2].z);
		glVertex3f(Boards[i].PointList[3].x, Boards[i].PointList[3].y, Boards[i].PointList[3].z);
		glEnd();
		glFlush();
	}

}


//���Ƶ�������
void DrawScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//�����ɫ����
	SetCamera();//�������
	DrawBoards();//���Ƶذ�ͱ߿�
	Balls.UpdateBalls();
	Balls.DrawBalls();//���ºͻ���С��
	glutSwapBuffers();
}

//ȫ�ֶ�ʱ��
void OnTimer(int value)
{
	glutPostRedisplay();//��ǵ�ǰ������Ҫ���»��ƣ�����myDisplay()
	glutTimerFunc(20, OnTimer, 1);
}


//������������
//��������� 
void OnMouseClick(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		TheCamera.MouseDown(x, y);
	}
}

//��������϶�  
void OnMouseMove(int x, int y)
{
	TheCamera.MouseMove(x, y);
}

//������̵����WASD��
void OnKeyClick(unsigned char key, int x, int y)
{
	int type = -1;
	if (key == 'w')
	{
		type = 0;
	}
	else if (key == 'a')
	{
		type = 1;
	}
	else if (key == 's')
	{
		type = 2;
	}
	else if (key == 'd')
	{
		type = 3;
	}
	TheCamera.KeyboardMove(type);
}

//������̵����ǰ�����ң�
void OnSpecialKeyClick(GLint key, GLint x, GLint y)
{
	int type = -1;
	if (key == GLUT_KEY_UP)
	{
		type = 0;
	}
	if (key == GLUT_KEY_LEFT)
	{
		type = 1;
	}
	if (key == GLUT_KEY_DOWN)
	{
		type = 2;
	}
	if (key == GLUT_KEY_RIGHT)
	{
		type = 3;
	}
	TheCamera.KeyboardMove(type);
}


//reshape����
void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(75.0f, (float)w / h, 1.0f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	InitWindow();             //��ʼ������
	InitScene();              //��ʼ������
	glutReshapeFunc(reshape); //��reshape����
	glutDisplayFunc(DrawScene); //����ʾ����
	glutTimerFunc(20, OnTimer, 1);  //������ʱ��
	glutMouseFunc(OnMouseClick); //�����������
	glutMotionFunc(OnMouseMove); //������ƶ�����
	glutKeyboardFunc(OnKeyClick);//�󶨼��̵������
	glutSpecialFunc(OnSpecialKeyClick);//��������̵������
	glutMainLoop();
}
