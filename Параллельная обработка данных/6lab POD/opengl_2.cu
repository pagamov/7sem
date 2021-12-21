// nvcc opengl_2.cu -lGL -lGLU -lglut -lGLEW
// ./a.out

// Для linux нужно поставить пакеты: libgl1-mesa-dev libglew-dev freeglut3-dev
// sudo apt-get install libgl1-mesa-dev libglew-dev freeglut3-dev

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

typedef unsigned char uchar;

#define sqr3(x) ((x)*(x)*(x))
#define sqr(x) ((x)*(x))

struct t_item {
	float x;
	float y;
	float z;
	float dx;
	float dy;
	float dz;
	float q;
};

t_item item;

int w = 1024, h = 648;

float x = -1.5, y = -1.5, z = 1.0;
float dx = 0.0, dy = 0.0, dz = 0.0;
float yaw = 0.0, pitch = 0.0;
float dyaw = 0.0, dpitch = 0.0;

float speed = 0.05;

const float a2 = 15.0;			
const int np = 100;				// Размер текстуры пола

GLUquadric* quadratic;			// quadric объекты - это геометрические фигуры 2-го порядка, т.е. сфера, цилиндр, диск, конус. 

cudaGraphicsResource *res;		
GLuint textures[2];				// Массив из текстурных номеров
GLuint vbo;						// Номер буфера


__global__ void kernel(uchar4 *data, t_item item, float t) {	// Генерация текстуры пола на GPU
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	float x, y, fg, fb;
	for(i = idx; i < np; i += offsetx)
		for(j = idy; j < np; j += offsety) {
			x = (2.0 * i / (np - 1.0) - 1.0) * a2;
			y = (2.0 * j / (np - 1.0) - 1.0) * a2;
			fb = 100.0 * (sin(0.1 * x * x + t) + cos(0.1 * y * y + t * 0.6) + sin(0.1 * x * x + 0.1 * y * y + t * 0.3));
			fg = 10000.0 * item.q / (sqr(x - item.x) + sqr(y - item.y) + sqr(item.z) + 0.001);
			fg = min(max(0.0f, fg), 255.0f);
			fb = min(max(0.0f, fb), 255.0f);
			data[j * np + i] = make_uchar4(0, (int)fg, (int)fb, 255);
		}
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Задаем "объектив камеры"
	gluPerspective(90.0f, (GLfloat)w/(GLfloat)h, 0.1f, 100.0f);


	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Задаем позицию и направление камеры
	gluLookAt(x, y, z,
				x + cos(yaw) * cos(pitch),
				y + sin(yaw) * cos(pitch),
				z + sin(pitch),
				0.0f, 0.0f, 1.0f);

	glBindTexture(GL_TEXTURE_2D, textures[0]);	// Задаем текстуру


	static float angle = 0.0;
	glPushMatrix();
		glTranslatef(item.x, item.y, item.z);	// Задаем координаты центра сферы
		glRotatef(angle, 0.0, 0.0, 1.0);
		gluSphere(quadratic, 2.5f, 32, 32);
	glPopMatrix();
	angle += 0.15;


	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);	// Делаем активным буфер с номером vbo
	glBindTexture(GL_TEXTURE_2D, textures[1]);	// Делаем активной вторую текстуру
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)np, (GLsizei)np, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);	// Деактивируем буфер
	// Последний параметр NULL в glTexImage2D говорит о том что данные для текстуры нужно брать из активного буфера
	
	glBegin(GL_QUADS);			// Рисуем пол
		glTexCoord2f(0.0, 0.0);
		glVertex3f(-a2, -a2, 0.0);

		glTexCoord2f(1.0, 0.0);
		glVertex3f(a2, -a2, 0.0);

		glTexCoord2f(1.0, 1.0);
		glVertex3f(a2, a2, 0.0);

		glTexCoord2f(0.0, 1.0);
		glVertex3f(-a2, a2, 0.0);
	glEnd();

	
	glBindTexture(GL_TEXTURE_2D, 0);			// Деактивируем текстуру

	// Отрисовка каркаса куба				
	glLineWidth(2);								// Толщина линий				
	glColor3f(0.5f, 0.5f, 0.5f);				// Цвет линий
	glBegin(GL_LINES);							// Последующие пары вершин будут задавать линии
		glVertex3f(-a2, -a2, 0.0);
		glVertex3f(-a2, -a2, 2.0 * a2);

		glVertex3f(a2, -a2, 0.0);
		glVertex3f(a2, -a2, 2.0 * a2);

		glVertex3f(a2, a2, 0.0);
		glVertex3f(a2, a2, 2.0 * a2);

		glVertex3f(-a2, a2, 0.0);
		glVertex3f(-a2, a2, 2.0 * a2);
	glEnd();

	glBegin(GL_LINE_LOOP);						// Все последующие точки будут соеденены замкнутой линией
		glVertex3f(-a2, -a2, 0.0);
		glVertex3f(a2, -a2, 0.0);
		glVertex3f(a2, a2, 0.0);
		glVertex3f(-a2, a2, 0.0);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(-a2, -a2, 2.0 * a2);
		glVertex3f(a2, -a2, 2.0 * a2);
		glVertex3f(a2, a2, 2.0 * a2);
		glVertex3f(-a2, a2, 2.0 * a2);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);

	glutSwapBuffers();
}



void update() {
//	dz -= 0.0001;			// Гравитация

	float v = sqrt(dx * dx + dy * dy + dz * dz);
	if (v > speed) {		// Ограничение максимальной скорости
		dx *= speed / v;
		dy *= speed / v;
		dz *= speed / v;
	}
	x += dx; dx *= 0.99;
	y += dy; dy *= 0.99;
	z += dz; dz *= 0.99;
	if (z < 1.0) {			// Пол, ниже которого камера не может переместиться
		z = 1.0;
		dz = 0.0;
	}
	if (fabs(dpitch) + fabs(dyaw) > 0.0001) {	// Вращение камеры
		yaw += dyaw;
		pitch += dpitch;
		pitch = min(M_PI / 2.0 - 0.0001, max(-M_PI / 2.0 + 0.0001, pitch));
		dyaw = dpitch = 0.0;
	}

	float w = 0.9999, e0 = 1e-3, dt = 0.01, K = 50.0;

// Замедление
	item.dx *= w;
	item.dy *= w;
	item.dz *= w;

// Отталкивание от стен
	item.dx += item.q * item.q * K * (item.x - a2) / (sqr3(fabs(item.x - a2)) + e0) * dt;
	item.dx += item.q * item.q * K * (item.x + a2) / (sqr3(fabs(item.x + a2)) + e0) * dt;

	item.dy += item.q * item.q * K * (item.y - a2) / (sqr3(fabs(item.y - a2)) + e0) * dt;
	item.dy += item.q * item.q * K * (item.y + a2) / (sqr3(fabs(item.y + a2)) + e0) * dt;

	item.dz += item.q * item.q * K * (item.z - 2 * a2) / (sqr3(fabs(item.z - 2 * a2)) + e0) * dt;
	item.dz += item.q * item.q * K * (item.z + 0.0) / (sqr3(fabs(item.z + 0.0)) + e0) * dt;

// Отталкивание от камеры
	float l = sqrt(sqr(item.x - x) + sqr(item.y - y) + sqr(item.z - z));
	item.dx += 3.0 * item.q * K * (item.x - x) / (l * l * l + e0) * dt;
	item.dy += 3.0 * item.q * K * (item.y - y) / (l * l * l + e0) * dt;
	item.dz += 3.0 * item.q * K * (item.z - z) / (l * l * l + e0) * dt;

	item.x += item.dx * dt;
	item.y += item.dy * dt;
	item.z += item.dz * dt;

	static float t = 0.0;
	uchar4* dev_data;
	size_t size;
	cudaGraphicsMapResources(1, &res, 0);		// Делаем буфер доступным для CUDA
	cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res);	// Получаем указатель на память буфера
	kernel<<<dim3(32, 32), dim3(32, 8)>>>(dev_data, item, t);		
	cudaGraphicsUnmapResources(1, &res, 0);		// Возращаем буфер OpenGL'ю что бы он мог его использовать
	t += 0.01;

	glutPostRedisplay();	// Перерисовка
}

void keys(unsigned char key, int x, int y) {	// Обработка кнопок
	switch (key) {
		case 'w':                 // "W" Движение вперед
			dx += cos(yaw) * cos(pitch) * speed;
			dy += sin(yaw) * cos(pitch) * speed;
			dz += sin(pitch) * speed;
		break;
		case 's':                 // "S" Назад
			dx += -cos(yaw) * cos(pitch) * speed;
			dy += -sin(yaw) * cos(pitch) * speed;
			dz += -sin(pitch) * speed;
		break;
		case 'a':                 // "A" Влево
			dx += -sin(yaw) * speed;
			dy += cos(yaw) * speed;
			break;
		case 'd':                 // "D" Вправо
			dx += sin(yaw) * speed;
			dy += -cos(yaw) * speed;
		break;
		case 27:
			cudaGraphicsUnregisterResource(res);
			glDeleteTextures(2, textures);
			glDeleteBuffers(1, &vbo);
			gluDeleteQuadric(quadratic);
			exit(0);
		break;
	}
}

void mouse(int x, int y) {
	static int x_prev = w / 2, y_prev = h / 2;
	float dx = 0.005 * (x - x_prev);
    float dy = 0.005 * (y - y_prev);
	dyaw -= dx;
    dpitch -= dy;
	x_prev = x;
	y_prev = y;

	// Перемещаем указатель мышки в центр, когда он достиг границы
	if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) {
		glutWarpPointer(w / 2, h / 2);
		x_prev = w / 2;
		y_prev = h / 2;
    }
}

void reshape(int w_new, int h_new) {
	w = w_new;
	h = h_new;
	glViewport(0, 0, w, h);                                     // Сброс текущей области вывода
	glMatrixMode(GL_PROJECTION);                                // Выбор матрицы проекций
	glLoadIdentity();                                           // Сброс матрицы проекции
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(w, h);
	glutCreateWindow("OpenGL");

	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(keys);
	glutPassiveMotionFunc(mouse);
	glutReshapeFunc(reshape);

	glutSetCursor(GLUT_CURSOR_NONE);	// Скрываем курсор мышки

	int wt, ht;
	FILE *in = fopen("in.data", "rb");
	fread(&wt, sizeof(int), 1, in);
	fread(&ht, sizeof(int), 1, in);
	uchar *data = (uchar *)malloc(sizeof(uchar) * wt * ht * 4);
	fread(data, sizeof(uchar), 4 * wt * ht, in);
	fclose(in);

	glGenTextures(2, textures);
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
	// если полигон, на который наносим текстуру, меньше текстуры
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //GL_LINEAR);	// Интерполяция
	// если больше
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //GL_LINEAR);		
	

	quadratic = gluNewQuadric();
	gluQuadricTexture(quadratic, GL_TRUE);	

	glBindTexture(GL_TEXTURE_2D, textures[1]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	// Интерполяция 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	// Интерполяция	

	glEnable(GL_TEXTURE_2D);                             // Разрешить наложение текстуры
	glShadeModel(GL_SMOOTH);                             // Разрешение сглаженного закрашивания
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);                // Черный фон
	glClearDepth(1.0f);                                  // Установка буфера глубины
	glDepthFunc(GL_LEQUAL);                              // Тип теста глубины. 
	glEnable(GL_DEPTH_TEST);                			 // Включаем тест глубины
	glEnable(GL_CULL_FACE);                 			 // Режим при котором, тектуры накладываются только с одной стороны

	glewInit();						
	glGenBuffers(1, &vbo);								// Получаем номер буфера
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);			// Делаем его активным
	glBufferData(GL_PIXEL_UNPACK_BUFFER, np * np * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);	// Задаем размер буфера
	cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);				// Регистрируем буфер для использования его памяти в CUDA
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);			// Деактивируем буфер

	item.x = item.y = item.z = 5.0;						
	item.dx = item.dy = item.dz = 0.1;
	item.q = 1.0;

	glutMainLoop();
}
