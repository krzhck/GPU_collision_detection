#include<GL/glut.h>
#include<iostream>
#include<math.h>
#include<windows.h>
#include "coord.hpp"
#include "ball.hpp"
#include "light.hpp"
#include "camera.hpp"
#include "wall.hpp"
#include "ballset.hpp"
#include "collision.cuh"
using namespace std;

//ȫ�ֳ���
const int window_width = 800, window_height = 600, window_x = 100, window_y = 100;
const char window_name[] = "collision detection";
const float refresh_interval = 0.02; //ˢ��ʱ��
const int ball_cols = 5;
const float length = 10, width = 10, height = 20, max_radius = 1; //������X,Y,Z��Χ��-X,X),(0,H),(-Z,Z)

//���գ����
Camera camera0(20.0f, 10.0f);
Light light0;

//����
Wall walls[6]; //�߽�

BallSet balls(length, height, width, ball_cols, max_radius, refresh_interval);


//��ʼ����������
//��ʼ������
void InitWindow()
{
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(window_x, window_y);
	glutCreateWindow(window_name);

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
	glClearColor(light0.Color[0], light0.Color[1], light0.Color[2], light0.Color[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//���ù�Դ��Ϣ
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0.Ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0.Diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0.Specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light0.Position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	//������ȼ�⣬��ֻ������ǰ���һ��
	glEnable(GL_DEPTH_TEST);
}

//��ʼ���߽�͵ذ�
void InitWalls()
{
	//8����
	Coord bottomA(-length, 0, -width);
	Coord bottomB(-length, 0, width);
	Coord bottomC(length, 0, -width);
	Coord bottomD(length, 0, width);
	Coord topA(-length, height, -width);
	Coord topB(-length, height, width);
	Coord topC(length, height, -width);
	Coord topD(length, height, width);

	//���õذ�͵���λ��
	walls[0].InitPos(bottomA, bottomB, bottomD, bottomC); // bottom
	walls[1].InitPos(bottomA, bottomB, topB, topA);
	walls[3].InitPos(bottomC, bottomD, topD, topC);
	walls[2].InitPos(bottomA, bottomC, topC, topA);
	walls[4].InitPos(bottomB, bottomD, topD, topB);
	walls[5].InitPos(topA, topB, topD, topC); // top

	GLfloat color_bottom[4] = { 1.0, 1.0, 1.0 , 1};
	GLfloat ambient_bottom[4] = { 0.4, 0.4, 0.4 , 1};
	GLfloat diffuse_bottom[4] = { 0.4, 0.4, 0.4 , 1};
	GLfloat specular_bottom[4] = { 0.2, 0.2, 0.2 , 1};
	GLfloat shininess_bottom = 20;

	Shader shader_bottom(color_bottom, ambient_bottom, diffuse_bottom, specular_bottom, shininess_bottom);
	walls[0].InitColor(shader_bottom);

	//�������ܵ������
	GLfloat color_border[3] = { 1.0, 1.0, 1.0 };
	GLfloat ambient_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat diffuse_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat specular_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat shininess_border = 40;
	Shader shader_border(color_border, ambient_border, diffuse_border, specular_border, shininess_border);
	for (int i = 1; i < 5; i++)
	{
		walls[i].InitColor(shader_border);
	}
}

//��ʼ����������
void InitScene()
{

	InitLight();
	InitWalls();
	balls.InitBalls();
}

//���ƺ�������
//�������λ��
void SetCamera()
{
	glLoadIdentity();
	Coord camera_Pos = camera0.Pos;//������ӵ������  
	Coord camera_center = camera0.LookCenter;//�����ӵ���������
	gluLookAt(camera_Pos.x, camera_Pos.y, camera_Pos.z, camera_center.x, camera_center.y, camera_center.z, 0, 1, 0); //���ӵ㿴Զ��,y�᷽��(0,1,0)���Ϸ���  
}

//����2�߽�͵ذ�
void RenderWalls()
{
	for (int i = 0; i < 3; i++)
	{
		glColor3f(walls[i].WallShader.Color[0], walls[i].WallShader.Color[1], walls[i].WallShader.Color[2]);
		glMaterialfv(GL_FRONT, GL_AMBIENT, walls[i].WallShader.Ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, walls[i].WallShader.Diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, walls[i].WallShader.Specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, walls[i].WallShader.Shininess);


		glBegin(GL_POLYGON);
		glVertex3f(walls[i].Vertexes[0].x, walls[i].Vertexes[0].y, walls[i].Vertexes[0].z);
		glVertex3f(walls[i].Vertexes[1].x, walls[i].Vertexes[1].y, walls[i].Vertexes[1].z);
		glVertex3f(walls[i].Vertexes[2].x, walls[i].Vertexes[2].y, walls[i].Vertexes[2].z);
		glVertex3f(walls[i].Vertexes[3].x, walls[i].Vertexes[3].y, walls[i].Vertexes[3].z);
		glEnd();
		glFlush();
	}

}


//���Ƶ�������
void RenderScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//�����ɫ����
	SetCamera();//�������
	RenderWalls();//���Ƶذ�ͱ߿�
	balls.UpdateBalls();
	balls.RenderBalls();//���ºͻ���С��
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
		camera0.Mousebottom(x, y);
	}
}

//��������϶�  
void OnMouseMove(int x, int y)
{
	camera0.MouseMove(x, y);
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
	camera0.KeyboardMove(type);
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
	camera0.KeyboardMove(type);
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
	glutDisplayFunc(RenderScene); //����ʾ����
	glutTimerFunc(20, OnTimer, 1);  //������ʱ��
	glutMouseFunc(OnMouseClick); //�����������
	glutMotionFunc(OnMouseMove); //������ƶ�����
	glutKeyboardFunc(OnKeyClick);//�󶨼��̵������
	glutSpecialFunc(OnSpecialKeyClick);//��������̵������
	glutMainLoop();
}
