using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using Emgu.CV.XFeatures2D;
using Emgu.CV.CvEnum;

namespace projektLWKForms
{
    public partial class Form1 : Form
    {
        private Mat loadedImage { get; set; }
        private UMat originalImageDescriptors { get; set; }
        private VectorOfKeyPoint originalImageKeyPoints { get; set; }
        private Capture capture;
        private Mat grayImage;
        private int time;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            textBox1.Text = openFileDialog1.FileName;
            
        }

        private void button2_Click(object sender, EventArgs e)
        {
            loadImage(textBox1.Text);
            grayImage = new Mat();
            CvInvoke.CvtColor(loadedImage, grayImage, ColorConversion.Bgr2Gray);
            originalImageDescriptors = new UMat();

            originalImageKeyPoints = new VectorOfKeyPoint();
            calculatedescriptors(loadedImage, originalImageDescriptors, originalImageKeyPoints);

            videoCompute();
        }

        private void loadImage(string file)
        {
            loadedImage = CvInvoke.Imread(file, Emgu.CV.CvEnum.LoadImageType.AnyColor);
        }

        private void calculatedescriptors(Mat image, UMat imageDescriptors, VectorOfKeyPoint imageKeyPoints)
        {
            using (UMat mImage = image.ToUMat(Emgu.CV.CvEnum.AccessType.Read))
            {
                SIFT sift = new SIFT();
                sift.DetectAndCompute(mImage, null, imageKeyPoints, imageDescriptors, false);
            }
        }

        private void videoCompute()
        {
            capture = new Emgu.CV.Capture(0);
            timer1.Enabled=true;
            viewer.ShowDialog();

            time = 0;
        }

        private void viewer_Load(object sender, EventArgs e)
        {

        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            if (time == 10)
            {
                Mat frame = new Mat();
                capture.Retrieve(frame, 0);
                Mat grayVideo = new Mat();
                CvInvoke.CvtColor(frame, grayVideo, ColorConversion.Bgr2Gray);
                UMat videoDescriptors = new UMat();
                VectorOfKeyPoint videoKeyPoints = new VectorOfKeyPoint();
                calculatedescriptors(grayVideo, videoDescriptors, videoKeyPoints);
                VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();

                BFMatcher matcher = new BFMatcher(DistanceType.L2);
                matcher.Add(originalImageDescriptors);
                matcher.KnnMatch(videoDescriptors, matches, 2, null);
                Mat mask = mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                mask.SetTo(new MCvScalar(255));

                Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);
                Mat homography = new Mat();
                int nonZeroCount = CvInvoke.CountNonZero(mask);
                if (nonZeroCount >= 4)
                {
                    nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(originalImageKeyPoints, videoKeyPoints,
                       matches, mask, 1.5, 20);
                    if (nonZeroCount >= 4)
                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(originalImageKeyPoints, videoKeyPoints, matches, mask, 2);
                }

                Mat result = new Mat();
                Features2DToolbox.DrawMatches(grayImage, originalImageKeyPoints, grayVideo, videoKeyPoints,
                   matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask);
                if (homography != null)
                {
                    //draw a rectangle along the projected model
                    Rectangle rect = new Rectangle(Point.Empty, grayImage.Size);
                    PointF[] pts = new PointF[]
                    {
                              new PointF(rect.Left, rect.Bottom),
                              new PointF(rect.Right, rect.Bottom),
                              new PointF(rect.Right, rect.Top),
                              new PointF(rect.Left, rect.Top)
                    };
                    pts = CvInvoke.PerspectiveTransform(pts, homography);

                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                    using (VectorOfPoint vp = new VectorOfPoint(points))
                    {
                        CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                    }
                    viewer.Image = result;
                    
                }

                
                time = 0; 
            }
            else
            {
                time++;
            }
    }
    }
}
