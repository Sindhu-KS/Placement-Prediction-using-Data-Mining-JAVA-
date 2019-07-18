3.4 CODE 

import java.io.BufferedReader;
	import java.io.FileReader;
	import java.util.Random;

	import weka.classifiers.Evaluation;
	import weka.classifiers.bayes.NaiveBayes;
	import weka.core.Instances;

	public class StartWeka {

		public static void main(String[] args) throws Exception {
			BufferedReader breader = null;
			breader = new BufferedReader(new FileReader("C:\\placement.arff"));
			Instances train = new Instances(breader);
			train.setClassIndex(train.numAttributes()-1);
			breader.close();
			NaiveBayes nB=new NaiveBayes();
			nB.buildClassifier(train);
			weka.classifiers.Evaluation eval=new Evaluation(train);
			eval.crossValidateModel(nB,train,10,new Random(1));
			System.out.println(eval.toSummaryString("\nResults\n====\n",true));
			System.out.println(eval.fMeasure(1)+" "+eval.precision(1)+" "+eval.recall(1));
			

		}

	}


Code 2:
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.attribute.AddClassification;


public class ClassificationPrediction {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("C:\\placement.arff");
		Instances traindata = source.getDataSet();
		traindata.setClassIndex(traindata.numAttributes()-1);
		int numClasses = traindata.numClasses();
		for (int i=0;i<numClasses;i++){
			String classValue = traindata.classAttribute().value(i);
			System.out.println("the "+i+"th class value:"+classValue);
		}
		/**
		 * naive bayes classifier	
		 */
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(traindata);
		/**
		 * load test data
		 */
		DataSource source2 = new DataSource("C:\\placement.arff");
		Instances testdata = source2.getDataSet();
		testdata.setClassIndex(testdata.numAttributes()-1);
		
		
		/**
		 * make prediction by naive bayes classifier
		 */
		for (int j=0;j<testdata.numInstances();j++){
			double actualClass = testdata.instance(j).classValue();
			String actual = testdata.classAttribute().value((int) actualClass);
			Instance newInst = testdata.instance(j);
			System.out.println("actual class:"+newInst.stringValue(newInst.numAttributes()-1));
			double preNB = nb.classifyInstance(newInst);
			String predString = testdata.classAttribute().value((int) preNB);
			System.out.println(actual+","+predString);
		}
		
		
	}

}

